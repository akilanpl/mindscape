"""Native engineering GPU preflight: real graph backward, zero optimizer updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def block_optimizers(torch):
    # A preflight never constructs or steps an optimizer, even on a native host.
    def forbidden(*args, **kwargs):
        raise RuntimeError('Optimizer construction and updates forbidden in engineering preflight')
    optimizer_base = torch.optim.Optimizer
    for name in dir(torch.optim):
        value = getattr(torch.optim, name)
        if isinstance(value, type) and issubclass(value, optimizer_base):
            value.__init__ = forbidden
            value.step = forbidden


def run():
    import torch
    block_optimizers(torch)
    receipt = {'kind': 'native-mps-engineering-preflight-v1', 'scientific_qualification': False,
               'optimizer_updates': 0, 'python': sys.version, 'executable': sys.executable,
               'os': platform.platform(), 'machine': platform.machine(),
               'torch': str(torch.__version__), 'sandbox': os.environ.get('CODEX_SANDBOX'),
               'mps_built': torch.backends.mps.is_built(),
               'mps_available_in_this_context': torch.backends.mps.is_available(),
               'fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'), 'measurements': []}
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None, '0'):
        raise RuntimeError('MPS fallback must be disabled before Python starts')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or not torch.backends.mps.is_available():
        receipt.update(status='BLOCKED_CURRENT_CONTEXT', explanation='This context cannot qualify native MPS. A sandbox result is not hardware absence.')
        return receipt
    from mindscape.r56.neural import R56Model
    from mindscape.r56.native_domains import pack_problem
    from mindscape.r56.benchmarks import matched_baseline, controlled_graph, model_accounting
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    allocation = torch.ones(16, device='mps')
    assert allocation.device.type == 'mps'
    problems = [
        {'domain':'math','operation':'add_digit','left':2,'right':3,'carry_in':0,'concept_ids':[0]},
        {'domain':'coding','operation':'rename','source':'x = 1\ny = x + x','old_name':'x','candidates':['a','b','c','d'],'requested_name':'c','concept_ids':[3]},
        {'domain':'physics','operation':'force','mass':2.,'force':[2.,-4.],'dt':.5,'velocity':[1.,-1.],'concept_ids':[2]}]
    for control in ('graph-concept-on','graph-concept-off','matched-gru'):
        torch.manual_seed(17)
        model = (matched_baseline('gru', seed=17) if control == 'matched-gru' else R56Model()).to('mps')
        before = {key: value.detach().cpu().clone() for key,value in model.state_dict().items()}
        for problem in problems:
            graph = controlled_graph(pack_problem(problem).to('mps'), control)
            times=[]
            for iteration in range(4):
                model.zero_grad(set_to_none=True)
                torch.mps.synchronize(); began=time.perf_counter()
                pred=model(graph, memory=None, stateless=False)
                if problem['domain']=='math':
                    loss=torch.nn.functional.cross_entropy(pred['math_digit_logits'][None],torch.tensor([5],device='mps'))+torch.nn.functional.cross_entropy(pred['math_carry_logits'][None],torch.tensor([0],device='mps'))
                elif problem['domain']=='coding':
                    loss=torch.nn.functional.cross_entropy(pred['coding_logits'][:4][None],torch.tensor([2],device='mps'))
                else:
                    loss=torch.nn.functional.mse_loss(pred['physics_delta'],torch.tensor([.5,-1.],device='mps'))
                assert loss.device.type=='mps' and bool(torch.isfinite(loss))
                loss.backward(); torch.mps.synchronize()
                elapsed=time.perf_counter()-began
                gradients=[p.grad for p in model.parameters() if p.grad is not None]
                assert gradients and all(g.device.type=='mps' and bool(torch.isfinite(g).all()) for g in gradients)
                assert all(p.device.type=='mps' for p in model.parameters())
                if iteration: times.append(elapsed)
            receipt['measurements'].append({'control':control,'domain':problem['domain'],
                'forward_backward_seconds':times,'device':'mps','loss_device':str(loss.device),
                'gradient_devices':sorted({str(g.device) for g in gradients}),
                'mps_current_allocated_bytes':torch.mps.current_allocated_memory(),
                'mps_driver_allocated_bytes':torch.mps.driver_allocated_memory(),**model_accounting(model)})
        assert all(torch.equal(value.detach().cpu(),before[key]) for key,value in model.state_dict().items())
    receipt.update(status='ENGINEERING_PASSED_NOT_SCIENTIFICALLY_QUALIFIED',weights_unchanged=True,
                   state_policy='memory reset each record; stateless=False',deterministic_algorithms=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve original preflight receipt')
    try:
        receipt=run()
    except Exception as exc:
        receipt={'status':'FAILED_ENGINEERING_PREFLIGHT','error_type':type(exc).__name__,
                 'error':str(exc),'optimizer_updates':0,'scientific_qualification':False,
                 'python':sys.version,'executable':sys.executable,'os':platform.platform(),
                 'machine':platform.machine(),'sandbox':os.environ.get('CODEX_SANDBOX'),
                 'fallback':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}
        try:
            import torch
            receipt.update(torch=str(torch.__version__),mps_built=torch.backends.mps.is_built(),
                           mps_available_in_this_context=torch.backends.mps.is_available())
        except ImportError:
            pass
    receipt['preflight_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':receipt['status'],'receipt':str(args.output),'optimizer_updates':0}))
    sys.exit(0 if receipt['status'].startswith('ENGINEERING_PASSED') else 2)
