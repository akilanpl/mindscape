"""One neural asset slot; CPU tools do not import or load neural models."""
import gc,sys
from .evidence import asset

class Resources:
    def __init__(self):self.key=None;self.value=None;self.loads=[]
    def close(self):
        self.value=None;self.key=None;gc.collect()
        if 'torch' in sys.modules:
            torch=sys.modules['torch']
            if torch.backends.mps.is_available():torch.mps.empty_cache()
    def slot(self,key,loader):
        if key!=self.key:
            self.close();self.value=loader();self.key=key;self.loads.append(key)
        return self.value
    def math(self):
        def load():
            import torch
            from mindscape.v3.physics_world.accelerator import require_mps,verify_placement
            from mindscape.r5_2_math.model import NumericTransition
            require_mps();path,row=asset('math_neural');saved=torch.load(path,map_location='cpu',weights_only=True)
            if (saved['seed'],saved['kind'],saved['updates'])!=(11,'recurrent',800):raise ValueError('Pinned arithmetic checkpoint identity')
            with torch.device('mps'):model=NumericTransition(11)
            model.load_state_dict(saved['state']);model.eval();verify_placement(model)
            return model,row
        model,row=self.slot('math_neural',load)
        from mindscape.r5_2_math.model import Neural
        return Neural(model),{'method':'R5.2_recurrent_experimental','neural':True,'shared_parameters':False,'checkpoint_sha256':row['sha256'],'scientific_status':'NO_GO; post-hoc recovery','selection':'fixed seed11 final800; no accuracy selection'}
    def physics(self):
        def load():
            import torch
            from mindscape.r5_3_physics.model import World
            path,row=asset('physics_neural');saved=torch.load(path,map_location='cpu',weights_only=True)
            if (saved['seed'],saved['kind'],saved['updates'])!=(11,'structured',160):raise ValueError('Pinned physics checkpoint identity')
            model=World(11,'structured');model.load_state_dict(saved['state']);model.eval()
            return model,row
        model,row=self.slot('physics_neural',load)
        return model,{'method':'R5.3_structured_experimental','neural':True,'shared_parameters':False,'checkpoint_sha256':row['sha256'],'scientific_status':'NO_GO; post-hoc recovery','selection':'fixed seed11 final160; no accuracy selection'}
    def shared(self):
        def load():
            import torch
            from mindscape.v5.model import GraphTransition
            from mindscape.v3.physics_world.accelerator import require_mps,verify_placement
            require_mps();path,row=asset('shared_core');saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['identity']!=row['identity'] or saved['step']!=120:raise ValueError('Shared checkpoint identity')
            with torch.device('mps'):model=GraphTransition(11)
            model.load_state_dict(saved['model']);model.eval();verify_placement(model)
            if sum(p.numel() for p in model.parameters())!=6967:raise ValueError('Shared architecture changed')
            return model,row
        return self.slot('shared_core',load)
    def coding(self,request,private):
        def load():
            from mindscape.v3.physics_world.accelerator import require_mps
            from mindscape.v3.integration.assets import Assets,QWEN,TRAINING,WASI
            require_mps()
            # Use the original coding loader/method unchanged; avoid unrelated math/physics preflight and loading.
            result=Assets.__new__(Assets);result.private={};result.loaded={};result.hashes={}
            for name,key in [('qwen',QWEN/'model.safetensors'),('coding_evaluator',TRAINING/'evaluator-s11.pt'),('coding_memory',TRAINING/'memory.json'),('wasi',WASI/'python.wasm'),('wasi_compiled',WASI/'python.cwasm')]:
                path,row=asset(name)
                if path.resolve()!=key.resolve():raise ValueError('Preserved coding asset location changed')
                result.hashes[str(key)]=row['sha256']
            return result
        loaded=self.slot('coding',load);loaded.private={request.identity:private}
        return loaded.coding(request)
