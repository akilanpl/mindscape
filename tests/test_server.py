import json,threading,urllib.request,urllib.error
import pytest
from mindscape_delivery.server import MindServer
from mindscape_delivery.assistant import Assistant

@pytest.fixture
def local_server():
    server=MindServer(('127.0.0.1',0),orchestrator=Assistant())
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield server
    server.shutdown();server.server_close();thread.join(3)

def test_full_http_question_and_assets(local_server):
    base=f'http://127.0.0.1:{local_server.server_address[1]}'
    assert b'Mindscape' in urllib.request.urlopen(base+'/').read()
    health=json.load(urllib.request.urlopen(base+'/api/health'))
    assert health['neural']['ready'] is False
    data=json.dumps({'message':'Differentiate x^2 with respect to x','inference_mode':'tools'}).encode()
    req=urllib.request.Request(base+'/api/chat',data=data,headers={'Content-Type':'application/json'})
    result=json.load(urllib.request.urlopen(req))
    assert result['solver_result']['answer']=='2*x'
    assert result['status']=='verified'

def test_host_origin_and_json_validation(local_server):
    base=f'http://127.0.0.1:{local_server.server_address[1]}'
    req=urllib.request.Request(base+'/api/health',headers={'Host':'evil.example'})
    with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(req)
    assert e.value.code==403
    req=urllib.request.Request(base+'/api/chat',data=b'{}',headers={'Content-Type':'application/json','Origin':'https://evil.example'})
    with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(req)
    assert e.value.code==403

def test_api_native_mode_does_not_run_pretrained(local_server):
    base=f'http://127.0.0.1:{local_server.server_address[1]}'
    data=json.dumps({'message':'Integrate x^2','mode':'native'}).encode()
    req=urllib.request.Request(base+'/api/chat',data=data,headers={'Content-Type':'application/json'})
    assert json.load(urllib.request.urlopen(req))['method']=='native R6 unsupported input'
