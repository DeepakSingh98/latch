"""Exact frozen top-k component functions; test-only, no cloud harness dependency."""
def component_forward(_parameters,scores):
    from megakernel_lab.operation_compiler.torch_topk32 import topk32
    values,indices=topk32(scores,8)
    return dict(values=values,indices=indices)


def component_ad(_parameters,scores,cotangent):
    import torch
    from megakernel_lab.operation_compiler.torch_topk32 import topk32
    def objective(x):
        values,indices=topk32(x,8)
        return (values*cotangent).sum(),(values,indices)
    gradient,(_,aux)=torch.func.grad_and_value(objective,has_aux=True)(scores)
    return dict(values=aux[0],indices=aux[1],gradient=gradient)


