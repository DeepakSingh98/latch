"""Explicit serial BF16 matrix epilogue contract for recognized readout VJPs.

This opt-in is an arithmetic choice, never a shape-selected optimization. It
binds a caller-supplied K partition to the derivative of an observed BF16
linear/log-softmax head. Every partition accumulates from zero in FP32, adds
the previous BF16 destination widened to FP32, and rounds once back to BF16.
The numerical suitability of a partition size must be measured against the
caller's reference implementation; no cuBLAS heuristic is inferred here.
"""
from __future__ import annotations

import copy

from .ir import CaptureError

KEY = 'serial_bf16_readout'
CONTRACT = 'bf16-serial-partition-epilogue-v1'
MMA_K8_CONTRACT = 'bf16-mma-k8-readout-serial-v1'
MMA_K8_RESIDUE_CONTRACT = 'bf16-mma-k8-readout-initial-residue-v1'


def _weight_plan(data, plan):
    """Prove the paired dWeight product and its sole widening consumer."""
    from . import vocab_head as head
    producer, consumers = head._index(data)
    values, matches = data['values'], []
    def output_strings(item):
        if isinstance(item,str):
            return {item}
        if isinstance(item,dict):
            return set().union(*(output_strings(x) for x in item.values()))
        if isinstance(item,(list,tuple)):
            return set().union(*(output_strings(x) for x in item))
        return set()
    exported = output_strings(data.get('outputs',{}))
    exported.update(output_strings(data.get('state_writes',{})))
    exported.update(data.get('state_writes',{}))
    exported.update(name for name,value in values.items() if value.get('output_roles'))
    for op in data['operations']:
        if op['target'] != 'aten.mm.default':
            continue
        a, b = (head._ref(arg) for arg in op['args'][:2])
        if a is None or b is None:
            continue
        paired = ((head._view_chain(a, producer, plan['dlogits']) is not None
                   and head._view_chain(b, producer, plan['a']) is not None)
                  or (head._view_chain(a, producer, plan['a']) is not None
                      and head._view_chain(b, producer, plan['dlogits']) is not None))
        if not paired:
            continue
        m, k = values[a]['shape']
        k2, n = values[b]['shape']
        output = op['outputs'][0]
        if (k != plan['rows'] or k2 != k or list(values[output]['shape']) != [m,n]
                or any(values[x]['dtype'] != 'bfloat16' for x in (a,b,output))):
            continue
        current, views = output, []
        while True:
            try:
                user = head._only(consumers, current, exported)
            except head.Refused as error:
                raise CaptureError('SERIAL_BF16', 'dWeight BF16 value escapes before widening: '+str(error)) from error
            if (user['target'] == 'aten._to_copy.default'
                    and values[user['outputs'][0]]['dtype'] == 'float32'
                    and list(values[current]['shape']) == list(values[user['outputs'][0]]['shape'])):
                break
            if user['target'] not in head._TRANSPOSING_VIEWS:
                raise CaptureError('SERIAL_BF16', 'dWeight needs one exclusive FP32 widening consumer')
            current = user['outputs'][0]
            views.append(current)
        matches.append(dict(operation=op, a=a, b=b, output=output, shape=[m,n,k],
                            views=views, cast=user['outputs'][0], head=plan))
    if len(matches) != 1:
        raise CaptureError('SERIAL_BF16', f'paired readout needs exactly one dWeight product, found {len(matches)}')
    return matches[0]


def _chunk_weight(data, plan, chunk_rows, products):
    """Ordinary ATen rewrite: BF16 partials, then ascending FP32 additions.

    The original BF16 product may only reach its single FP32 cast through
    views. We promote that private chain to FP32, preserving every external
    SSA name. Serialized ATen replay therefore expresses the actual rounding
    contract rather than relying on an opaque matrix-output dtype override.
    """
    from . import vocab_head as head
    values, output = data['values'], plan['output']
    m,n,k = plan['shape']
    original, made, partials, widened_partials, additions, intervals = plan['operation'], [], [], [], [], []
    inputs = set(data['inputs'])
    def add(name, target, args, shape, dtype, *, alias=None, strides=None, offset=0, kwargs=None):
        head._new_value(data,name,shape,dtype,alias_of=alias,strides=strides,offset=offset)
        op = head._new_op(name,target,args,kwargs,view=alias is not None)
        op['dependencies'] = [x for x in op['inputs'] if x not in inputs]
        made.append(op)
    total = None
    for index,start in enumerate(range(0,k,chunk_rows)):
        stop = min(start+chunk_rows,k)
        stem = f'serial_weight_{output}_{index}'
        left, right, partial, widened = (stem+s for s in ('_a','_b','_mm','_f32'))
        for name,source,dim,shape in ((left,plan['a'],1,[m,stop-start]),
                                       (right,plan['b'],0,[stop-start,n])):
            value = values[source]
            add(name,'aten.slice.Tensor',[{'ref':source},dim,start,stop,1],shape,'bfloat16',
                alias=source,strides=value['strides'],
                offset=value['storage_offset']+start*value['strides'][dim])
        add(partial,'aten.mm.default',[{'ref':left},{'ref':right}],[m,n],'bfloat16')
        add(widened,'aten._to_copy.default',[{'ref':partial}],[m,n],'float32',kwargs={'dtype':'float32'})
        products[partial] = dict(a=left,b=right,shape=[m,n,stop-start],partition_k=0,
            dtype='bfloat16',contract=MMA_K8_CONTRACT,role='weight_partial',
            forward_product=plan['head']['product'],dlogits=plan['head']['dlogits'],
            weight_root=plan['head']['root'],chunk_rows=chunk_rows,token_interval=[start,stop],
            proof='paired head dWeight; exclusive BF16-to-FP32 widening; ordered token slices')
        partials.append(partial)
        widened_partials.append(widened)
        intervals.append([start,stop])
        if total is None:
            total = widened
        else:
            next_total = stem+'_sum'
            add(next_total,'aten.add.Tensor',[{'ref':total},{'ref':widened}],[m,n],'float32')
            additions.append(next_total)
            total = next_total
    # Preserve the original definition name and all downstream names. The
    # final identity copy is explicit and its replay dtype is now FP32.
    tail = head._new_op(output,'aten._to_copy.default',[{'ref':total}],{'dtype':'float32'})
    tail['name'] = original['name']
    tail['dependencies'] = [total]
    made.append(tail)
    for name in [output,*plan['views']]:
        values[name]['dtype'] = 'float32'
    data['operations'] = [item for op in data['operations']
                          for item in (made if op is original else [op])]
    return dict(product=output,cast=plan['cast'],chunk_rows=chunk_rows,
                partials=partials,widened_partials=widened_partials,
                subtotal_additions=additions,token_intervals=intervals,
                accumulation='ascending FP32 addition of individually BF16-rounded partials')


def _weight_accumulation(data, plans, chunks):
    """Flatten only a proved private head-add tree, preserving its outer users.

    The caller explicitly requests last-forward-first, followed by the other
    heads in forward order. The first head's ascending subtotal is an exact
    prefix of that sequence. Subsequent individual widened partials are added
    directly, without intermediate head subtotals. No operation is rescheduled.
    """
    from . import vocab_head as head
    producer, consumers = head._index(data)
    positions = {op['outputs'][0]:i for i,op in enumerate(data['operations'])}
    values = data['values']
    def output_strings(item):
        if isinstance(item,str): return {item}
        if isinstance(item,dict): return set().union(*(output_strings(x) for x in item.values()))
        if isinstance(item,(list,tuple)): return set().union(*(output_strings(x) for x in item))
        return set()
    exported = output_strings(data.get('outputs',{}))
    exported.update(output_strings(data.get('state_writes',{})))
    exported.update(data.get('state_writes',{}))
    exported.update(n for n,v in values.items() if v.get('output_roles'))
    def parameter_root(name):
        seen=set()
        while name in producer:
            if name in seen: raise CaptureError('SERIAL_BF16','cyclic shared-weight ancestry')
            seen.add(name);op=producer[name]
            if op['target'] not in head._TRANSPOSING_VIEWS|{'aten._to_copy.default'}:
                raise CaptureError('SERIAL_BF16','shared weight is not a transparent parameter view/cast')
            name=head._ref(op['args'][0])
        if values.get(name,{}).get('role')!='parameter':
            raise CaptureError('SERIAL_BF16','shared head weight must originate at one parameter')
        return name
    groups={}
    for plan,chunk in zip(plans,chunks):
        if plan['views']:
            raise CaptureError('SERIAL_BF16','flat weight accumulation requires direct dWeight widening')
        groups.setdefault(parameter_root(plan['head']['root']),[]).append((plan,chunk))
    reports=[];replacement={};remove=set()
    for parameter,members in groups.items():
        ordered=sorted(members,key=lambda pair:positions[pair[0]['head']['product']])
        ordered=ordered[-1:]+ordered[:-1]
        casts={p['cast'] for p,c in members}
        shape=values[parameter]['shape']
        if any(values[n]['shape']!=shape or values[n]['dtype']!='float32' for n in casts):
            raise CaptureError('SERIAL_BF16','flat head gradients must match their FP32 master shape')
        base=dict(parameter_root=parameter,order='last-forward-first',
            forward_products=[p['head']['product'] for p,c in ordered],
            partials=[n for p,c in ordered for n in c['partials']],
            widened_partials=[n for p,c in ordered for n in c['widened_partials']],
            preserves_outer_users=True,operation_rescheduling=False)
        if len(members)==1:
            reports.append({**base,'output':ordered[0][0]['cast'],'single_head_unchanged':True,
                'additional_retained_partial_bytes_upper_bound':0})
            continue
        memo={}
        def subtree(name):
            if name in memo:return memo[name]
            if name in casts:return ({name},set())
            op=producer.get(name)
            result=None
            if (op and op['target']=='aten.add.Tensor' and len(op['args'])==2
                    and op.get('kwargs',{}) in ({},{'alpha':1})
                    and values[name]['dtype']=='float32' and values[name]['shape']==shape):
                a,b=(head._ref(x) for x in op['args'])
                left,right=subtree(a),subtree(b)
                if left is not None and right is not None and not(left[0]&right[0]):
                    result=(left[0]|right[0],left[1]|right[1]|{name})
            memo[name]=result
            return result
        roots=[n for n in producer if (tree:=subtree(n)) is not None and tree[0]==casts]
        if len(roots)!=1:
            raise CaptureError('SERIAL_BF16','shared heads need one exclusive FP32 add tree without other leaves')
        root=roots[0];tree=subtree(root)[1]
        for name in casts|(tree-{root}):
            try:user=head._only(consumers,name,exported)
            except head.Refused as error:
                raise CaptureError('SERIAL_BF16','shared head subtotal escapes: '+str(error)) from error
            if user['outputs'][0] not in tree:
                raise CaptureError('SERIAL_BF16','shared head subtotal has an unrelated consumer')
        total=ordered[0][0]['cast'];made=[]
        flat_after_prefix=[n for p,c in ordered[1:] for n in c['widened_partials']]
        for index,partial in enumerate(flat_after_prefix):
            name=root if index==len(flat_after_prefix)-1 else f'serial_flat_{root}_{index}'
            if name!=root:head._new_value(data,name,shape,'float32')
            op=head._new_op(name,'aten.add.Tensor',[{'ref':total},{'ref':partial}])
            op['dependencies']=list(op['inputs']);made.append(op);total=name
        replacement[root]=made;remove.update(tree-{root})
        # Private non-prefix subtotals are no longer used. BF16 products and
        # their individual FP32 casts remain ordinary inspectable ATen nodes.
        for plan,chunk in ordered[1:]:
            remove.update(chunk['subtotal_additions'])
            remove.update((plan['output'],plan['cast']))
            chunk['source_only_removed_subtotals']=[plan['output'],plan['cast'],*chunk['subtotal_additions']]
        retained=sum(values[n]['shape'][0]*values[n]['shape'][1]*4 for n in flat_after_prefix)
        reports.append({**base,'output':root,'single_head_unchanged':False,
            'prefix_subtotal':ordered[0][0]['cast'],'additions':[op['outputs'][0] for op in made],
            'additional_retained_partial_bytes_upper_bound':retained,
            'source_subtotals':[p['cast'] for p,c in members]})
    data['operations']=[item for op in data['operations'] if op['outputs'][0] not in remove
        for item in replacement.get(op['outputs'][0],[op])]
    for name in remove:values.pop(name)
    return reports


def bind_readout_contract(ir, partition_k, *, instruction='wgmma-k16', chunk_rows=None, residue_first=False,
                          weight_accumulation=None):
    """Return a copied IR and graph-bound arithmetic provenance.

    Recognition uses only the existing head dataflow proof, dtypes, shapes,
    and identity of the shared forward/backward weight. Parameter/module names
    are not dispatch conditions. Padded vocabulary rewrites are disallowed by
    the calling compiler API because they change the observed contraction.
    """
    from . import vocab_head as head
    if type(partition_k) is not int or partition_k <= 0 or partition_k % 16:
        raise ValueError('serial BF16 partition_k must be a positive multiple of16')
    if instruction not in ('wgmma-k16', 'mma-k8'):
        raise ValueError('unknown readout MMA arithmetic instruction')
    if type(residue_first) is not bool or (residue_first and instruction != 'mma-k8'):
        raise ValueError('residue_first needs a bool and mma-k8 instruction')
    if chunk_rows is not None and (type(chunk_rows) is not int or chunk_rows <= 0
                                   or chunk_rows % 8 or instruction != 'mma-k8'):
        raise ValueError('chunk_rows needs a positive multiple of8 and mma-k8 instruction')
    if weight_accumulation not in (None,'last-forward-first') or (weight_accumulation and chunk_rows is None):
        raise ValueError('weight_accumulation needs last-forward-first and explicit chunk_rows')
    contract = MMA_K8_CONTRACT if instruction == 'mma-k8' else CONTRACT
    data = ir.to_dict()
    if KEY in data.get('metadata', {}):
        raise CaptureError('SERIAL_BF16', 'an arithmetic contract is already bound')
    producer, _ = head._index(data)
    values = data['values']
    products, refusals, weight_plans = {}, [], []
    for operation in data['operations']:
        if operation['target'] != 'aten._log_softmax.default':
            continue
        try:
            plan = head._match(data, operation)
        except head.Refused as error:
            refusals.append(dict(operation=operation['name'], reason=str(error)))
            continue
        if plan is None or not plan.get('back') or plan['dtype'] != 'bfloat16':
            continue
        # Restrict to the input VJP dLogits @ W, excluding the transposed
        # dWeight product. Direct gradient/weight views must preserve shape.
        for product in data['operations']:
            if product['target'] != 'aten.mm.default':
                continue
            a, b = (head._ref(arg) for arg in product['args'][:2])
            if a is None or b is None:
                continue
            if head._view_chain(a, producer, plan['dlogits']) is None:
                continue
            if head._view_chain(b, producer, plan['root']) is None:
                continue
            m, k = values[a]['shape']
            k2, n = values[b]['shape']
            output = product['outputs'][0]
            if ([m, k] != [plan['rows'], plan['vocab']] or k2 != k
                    or list(values[output]['shape']) != [m, n]
                    or any(values[name]['dtype'] != 'bfloat16' for name in (a, b, output))):
                continue
            if k <= partition_k:
                raise CaptureError('SERIAL_BF16', 'bound head needs more than one K partition')
            if output in products:
                raise CaptureError('SERIAL_BF16', 'one derivative matched multiple heads')
            products[output] = dict(a=a, b=b, shape=[m, n, k], partition_k=partition_k,
                                    dtype='bfloat16', contract=contract,
                                    forward_product=plan['product'], log_softmax=operation['name'],
                                    dlogits=plan['dlogits'], weight_root=plan['root'],
                                    proof='paired head dataflow and shared weight identity')
            if instruction == 'mma-k8':
                products[output]['role'] = 'input_vjp'
                if residue_first:
                    products[output].update(contract=MMA_K8_RESIDUE_CONTRACT,
                        residue_tile_k=8, k_grouping='initial-short-then-full-k8')
                forward = producer[plan['product']]
                fa, fb = (head._ref(arg) for arg in forward['args'][:2])
                fm, fk = values[fa]['shape']
                fk2, fn = values[fb]['shape']
                if (forward['target'] != 'aten.mm.default' or fk != fk2
                        or [fm, fn] != [plan['rows'], plan['vocab']]
                        or any(values[name]['dtype'] != 'bfloat16' for name in (fa,fb,plan['product']))):
                    raise CaptureError('SERIAL_BF16', 'paired forward BF16 product contract changed')
                products[plan['product']] = dict(a=fa, b=fb, shape=[fm,fn,fk],
                    partition_k=0, dtype='bfloat16', contract=contract, role='forward',
                    forward_product=plan['product'], log_softmax=operation['name'],
                    dlogits=plan['dlogits'], weight_root=plan['root'],
                    proof='paired head dataflow and shared weight identity')
                if chunk_rows is not None:
                    weight_plans.append(_weight_plan(data,plan))
    if not products:
        raise CaptureError('SERIAL_BF16', f'no admitted BF16 readout input VJP found; recognition refusals={refusals}')
    report = dict(contract=contract, partition_k=partition_k, products=products,
                  refused=refusals, qualification='caller must qualify complete workload',
                  arithmetic='independent FP32 K-partial from zero; add widened previous BF16; BF16 RN each partition')
    if instruction == 'mma-k8':
        report.update(instruction=instruction,
            arithmetic='forward: ascending mma.sync k8 from FP32 zero, one BF16 RN; input VJP: same k8 partials with explicit serial BF16 partition epilogues')
    if residue_first:
        report.update(contract=MMA_K8_RESIDUE_CONTRACT, residue_first=True)
        report['arithmetic'] += '; input VJP final partition: initial short k8 group zero-filled before full k8 groups'
    if chunk_rows is not None:
        if len({plan['output'] for plan in weight_plans}) != len(weight_plans):
            raise CaptureError('SERIAL_BF16','one dWeight product matched multiple heads')
        report['chunk_rows'] = chunk_rows
        report['weight_gradients'] = [_chunk_weight(data,plan,chunk_rows,products) for plan in weight_plans]
        report['arithmetic'] += '; weight VJP: BF16 partial products widened then added in ascending token order'
        if weight_accumulation:
            report['weight_accumulation'] = weight_accumulation
            report['weight_accumulation_groups'] = _weight_accumulation(data,weight_plans,report['weight_gradients'])
            report['arithmetic'] += '; shared master: last forward head then other heads in forward order, flat FP32 partial additions'
        data.setdefault('metadata',{})[KEY] = report
        bound = type(ir).from_dict(data)
    else:
        bound = copy.deepcopy(ir)
        bound.metadata[KEY] = report
    return bound, report


def region_contract(metadata, outputs):
    """Filter whole-graph bindings to products actually defined by a region."""
    record = metadata.get(KEY)
    if not record:
        return None
    outputs = set(outputs)
    selected = {name: item for name, item in record['products'].items() if name in outputs}
    return {**record, 'products': selected} if selected else None
