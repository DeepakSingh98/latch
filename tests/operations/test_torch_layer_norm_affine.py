"""Small metadata-only affine contracts; no tensor arithmetic or Torch import."""
from copy import deepcopy
import unittest

from .test_torch_layer_norm_backward import fixture as dx_fixture, ln


def fixture():
    graph = dx_fixture()
    for name,value in graph['values'].items():
        old=value['shape']
        shape=[16384,768] if old==[3,8] else [16384,1] if old==[3,1] else [768]
        value.update(name=name,shape=shape,strides=[shape[1],1] if len(shape)==2 else [1])
    for op in graph['operations']:
        if op['name'] in ('scale','scaled'):op['args'][1]=768
    for name,source in (('dgain','gain_product'),('dbias','dy')):
        graph['values'][name]=dict(name=name,shape=[768],strides=[1],dtype='float32',
                                  storage_offset=0,alias_of=None,output_roles=[])
        graph['operations'].append(dict(name=name,target='aten.sum.dim_IntList',
            args=[{'ref':source},[0]],kwargs={},inputs=[source],reads=[source],
            outputs=[name],writes=[name],effect='pure'))
    graph['outputs']=['output','dgain','dbias']
    graph['metadata'][ln.AFFINE_KEY]=ln.AFFINE_CONTRACT
    return graph


class LayerNormAffineMetadataTests(unittest.TestCase):
    def test_default_noop(self):
        g=fixture();g['metadata'].pop(ln.AFFINE_KEY);before=deepcopy(g)
        self.assertEqual(ln.find_affine_plans(g),[]);self.assertEqual(g,before)

    def test_primary_proof_and_nonmutation(self):
        g=fixture();before=deepcopy(g);p,=ln.find_affine_plans(g)
        self.assertEqual(p.members,('gain_product','dgain','dbias'))
        self.assertEqual(p.finals,('dgain','dbias'))
        self.assertEqual(p.intermediates,('gain_product',));self.assertEqual(g,before)
        self.assertFalse(set(p.members)&set(ln.find_plans(g)[0].members))

    def test_shared_gain_product_preserved(self):
        for kind in ('output','role','alias','read'):
            g=fixture()
            if kind=='output':g['outputs'].append('gain_product')
            elif kind=='role':g['values']['gain_product']['output_roles']=['saved']
            elif kind=='alias':g['values']['alias']=dict(g['values']['gain_product'],alias_of='gain_product')
            else:g['operations'].append(dict(name='reader',target='custom.read',args=[{'ref':'gain_product'}],
                inputs=['gain_product'],reads=['gain_product'],outputs=[],writes=[],kwargs={},effect='pure'))
            p,=ln.find_affine_plans(g)
            self.assertEqual(p.members,('dgain','dbias'));self.assertEqual(p.intermediates,())

    def test_semantic_refusals(self):
        for name,field,value in [('dgain','args',[{'ref':'gain_product'},[0],True]),
                                 ('dbias','kwargs',{'dtype':'float64'}),
                                 ('gain_product','kwargs',{'alpha':2}),
                                 ('dgain','effect','write')]:
            g=fixture();next(o for o in g['operations'] if o['name']==name)[field]=value
            self.assertEqual(ln.find_affine_plans(g),[])
            self.assertTrue(ln.analyze_affine(g)['refused'])

    def test_layout_and_saved_tape_refusals(self):
        for name,key,value in [('x','strides',[769,1]),('rstd','dtype','bfloat16'),
                                ('dgain','dtype','bfloat16'),('dbias','strides',[2])]:
            g=fixture();g['values'][name][key]=value
            self.assertEqual(ln.find_affine_plans(g),[])

    def test_unknown_policy(self):
        g=fixture();g['metadata'][ln.AFFINE_KEY]='unknown'
        with self.assertRaises(ValueError):ln.find_affine_plans(g)


if __name__=='__main__':unittest.main()
