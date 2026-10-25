# P-OPS2-A adaptation from 4dcee751fab59f5ecea67ebada3167ca93fba841.
"""Whole-program cooperative resource selection for the integrated primitives.

P-OPS-B's two-CTA execution setting is used only when every branch/loop region
fits the same conservative shared-memory ceiling. D attention (131840 bytes)
and large C stage rings retain one CTA per SM. The CUDA compiler enforces the
register bound, and the runtime always clamps to measured kernel occupancy.
This chooses resources, never a model and never arithmetic.
"""
from __future__ import annotations

TWO_CTA_SHARED_LIMIT = 98304
TWO_CTA_SHARED_MAX = TWO_CTA_SHARED_LIMIT  # accepted P6-A resource-interface alias


def select_program_resources(shared_bytes, *, phases=(), children=(), arch='sm_90', allow_two_cta=True,
                             pops2_two_cta=False, residency=None):
    if type(shared_bytes) is not int or not 0 < shared_bytes <= 232448:
        raise ValueError('invalid whole-program CUDA shared-memory requirement')
    if residency is not None and (type(residency) is not int or residency not in (1, 2)):
        raise ValueError('cooperative residency must be one or two CTAs per SM')
    if residency == 1:
        allow_two_cta = False
    engines, plan_grids = set(), set()
    def visit(node):
        if isinstance(node,(list,tuple)):
            for item in node:visit(item)
        elif isinstance(node,dict):
            engine=node.get('engine','')
            if isinstance(engine,str) and engine.startswith(('popsb','pops_b')):
                engines.add(engine)
                if 'selection_grid' in node:plan_grids.add(node['selection_grid'])
            resource=node.get('resource_policy',{})
            if resource.get('has_popsb'):
                engines.update(resource.get('popsb_engines',['popsb_cpasync']))
                plan_grids.update(resource.get('matrix_selection_grids',[]))
            for key in ('phases','items','members'):
                visit(node.get(key,()))
            for key in ('branches','regions'):
                visit(list(node.get(key,{}).values()))
    visit(phases);visit(list(children))
    has_b=bool(engines)
    # P-OPS2-A: products planned for two CTAs/SM request the same whole-program shared-memory proof.
    two = (has_b or pops2_two_cta) and allow_two_cta and arch=='sm_90' and shared_bytes<=TWO_CTA_SHARED_LIMIT
    reason = ('P-OPS-B whole-program shared ceiling admits the two-CTA register target' if two else
              'no P-OPS-B product; preserve existing resource policy' if not has_b else
              'whole-program shared memory exceeds the conservative two-CTA ceiling' if shared_bytes>TWO_CTA_SHARED_LIMIT else
              'explicit one-CTA resource choice or target outside H100 qualification')
    return {'version':'whole-program-resources-v1','resident_blocks_per_sm':2 if two else 1,
            'has_popsb':has_b,'pops2_two_cta':bool(pops2_two_cta),'popsb_engines':sorted(engines),'shared_bytes':shared_bytes,
            'two_cta_shared_limit':TWO_CTA_SHARED_LIMIT,'target_arch':arch,'reason':reason,
            'matrix_selection_grids':sorted(plan_grids),
            'register_admission':'CUDA launch_bounds(256,2)' if two else 'existing compiler register allocation',
            'runtime_admission':'actual CUDA occupancy query remains mandatory',
            'performance_scope':'integrated whole-update measurement required; donor ratios do not compose'}
