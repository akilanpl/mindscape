"""Engineering-only CPU equivalence and optional real MPS measurements."""
import json
import time
from dataclasses import replace
import torch
from mindscape.r56.neural import R56Model, GraphInput
from mindscape.r56.native_domains import pack_problem
from mindscape.r56.benchmarks import controlled_graph, matched_baseline
from scripts.native.original_neural_reference import R56Model as Original

def cases():
    problems=[{'domain':'math','left':7,'right':8,'carry_in':1,'concept_ids':[0]},
              {'domain':'coding','source':'def f(x):\n return x+x','old_name':'x','requested_name':'y','candidates':['x','y'],'concept_ids':[3]},
              {'domain':'physics','mass':2.,'force':[3.,-4.],'velocity':[1.,-2.],'dt':.5,'concept_ids':[2]}]
    for p in problems:
        g=pack_problem(p)
        yield p['domain'],g
        yield p['domain']+'-zero-edges',replace(g,edges=torch.empty(2,0,dtype=torch.long),relation_types=torch.empty(0,dtype=torch.long))
    yield 'boundary-256-nodes-2048-edges',GraphInput(torch.randn(256,8),torch.zeros(256,dtype=torch.long),
        torch.stack((torch.arange(2048)%256,torch.zeros(2048,dtype=torch.long))),torch.zeros(2048,dtype=torch.long),torch.zeros(4),torch.tensor([0]),0)
    yield 'boundary-one-node-zero-edges',GraphInput(torch.randn(1,8),torch.zeros(1,dtype=torch.long),torch.empty(2,0,dtype=torch.long),torch.empty(0,dtype=torch.long),torch.zeros(4),torch.tensor([0]),0)

def exercise(model,g):
    model.zero_grad(set_to_none=True)
    out=model(g,memory=None,stateless=False)
    loss=sum(out[k].square().mean() for k in ('math_digit_logits','math_carry_logits','coding_logits','physics_delta'))
    loss.backward()
    return out

def measure(device='cpu'):
    torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    records=[]
    for condition in ('graph-concept-on','graph-concept-off','matched-gru'):
        new=matched_baseline('gru') if condition=='matched-gru' else R56Model()
        old=matched_baseline('gru') if condition=='matched-gru' else Original()
        old.load_state_dict(new.state_dict())
        before={k:v.detach().clone() for k,v in new.state_dict().items()}
        for name,g in cases():
            g=controlled_graph(g,condition)
            # CPU original is the reference; never run nondeterministic MPS original.
            a=exercise(old,g);b=exercise(new,g)
            forward=max(float((a[k]-b[k]).detach().abs().max()) for k in a)
            grad=max(float((p.grad-q.grad).abs().max()) for p,q in zip(old.parameters(),new.parameters()) if p.grad is not None)
            for k in a:torch.testing.assert_close(a[k],b[k],atol=2e-5,rtol=2e-4)
            for p,q in zip(old.parameters(),new.parameters()):
                assert (p.grad is None)==(q.grad is None)
                if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-4)
            record={'condition':condition,'case':name,'cpu_forward_max_abs':forward,'cpu_gradient_max_abs':grad,'atol':2e-5,'rtol':2e-4}
            if device=='mps':
                gpu=matched_baseline('gru') if condition=='matched-gru' else R56Model()
                gpu.load_state_dict(new.state_dict());gpu.to('mps');gg=g.to('mps')
                gpu_before={k:v.detach().cpu().clone() for k,v in gpu.state_dict().items()}
                times=[]
                for i in range(4):
                    torch.mps.synchronize();start=time.perf_counter();out=exercise(gpu,gg);torch.mps.synchronize()
                    if i:times.append(time.perf_counter()-start)
                assert all(v.device.type=='mps' for v in out.values())
                assert all(p.grad.device.type=='mps' for p in gpu.parameters() if p.grad is not None)
                assert all(torch.equal(v.detach().cpu(),gpu_before[k]) for k,v in gpu.state_dict().items())
                record.update(mps_forward_backward_seconds=times,mps_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory(),incidence_bytes_upper_bound=2*len(g.node_features)*g.edges.shape[1]*4)
            assert all(torch.equal(v,before[k]) for k,v in new.state_dict().items())
            records.append(record)
    return {'records':records,'optimizer_updates':0,'weights_unchanged':True,'device':device,'deterministic':True}

if __name__=='__main__':print(json.dumps(measure(),indent=2))
