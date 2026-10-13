"""JSON-only immutable-by-copy request and provenance-bearing response contract."""
from dataclasses import dataclass, field, asdict
import copy, hashlib, json
VERSION='mindscape.kernel/1'
DOMAINS=('mathematics','coding','physics')

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def plain(value):
    import numpy as np
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value

@dataclass(frozen=True)
class Request:
    domain: str
    operation: str
    observation: dict
    action: dict=field(default_factory=dict)
    mode: str='validated'
    version: str=VERSION
    def __post_init__(self):
        if self.version!=VERSION or self.domain not in DOMAINS or self.mode not in ('validated','experimental_neural','experimental_shared'):
            raise ValueError('Unsupported contract version/domain/mode')
        if type(self.observation) is not dict or type(self.action) is not dict or type(self.operation) is not str:
            raise ValueError('Request field types')
        if len(canonical(asdict(self)))>200000:raise ValueError('Request size budget')
        object.__setattr__(self,'observation',copy.deepcopy(self.observation));object.__setattr__(self,'action',copy.deepcopy(self.action))
    @property
    def identity(self):return digest(asdict(self))

@dataclass(frozen=True)
class Response:
    request_id: str
    domain: str
    state: dict
    action: dict
    prediction: dict
    execution: dict
    verification: dict
    provenance: dict
    trace: tuple
    version: str=VERSION
    def __post_init__(self):
        if self.prediction.get('kind') not in ('neural_prediction','symbolic_result','procedural_prediction','submitted_program','no_prediction'):
            raise ValueError('Prediction provenance missing')
        if self.execution.get('kind') not in ('tool_result','actual_execution','not_executed'):
            raise ValueError('Execution provenance missing')
        if self.verification.get('success') is not None and type(self.verification['success']) is not bool:
            raise ValueError('Verification must be Boolean or untested')
        if self.provenance.get('learned_transfer') is not False:raise ValueError('Unsupported transfer claim')
        canonical(asdict(self))
    def record(self):return asdict(self)
