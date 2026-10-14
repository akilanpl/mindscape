import json,numpy as np
from mindscape.alpha_transfer.numerical_scoring import task_metrics
cases=[('nonbinary_valid',{'delta':[[1.]],'valid':2},{'delta':[[1.]],'valid_probability':1.}),('empty_graph',{'delta':np.zeros((0,1)).tolist(),'valid':1},{'delta':np.zeros((0,1)).tolist(),'valid_probability':1.}),('zero_coordinates',{'delta':[[]],'valid':1},{'delta':[[]],'valid_probability':1.})]
result=[]
for name,target,prediction in cases:
 try:
  v=task_metrics(prediction,target); result.append({'case':name,'rejected':False,'result':v})
 except Exception as e: result.append({'case':name,'rejected':True,'exception':str(e)})
print(json.dumps(result,indent=2))
