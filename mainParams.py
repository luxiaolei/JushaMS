"""
mainParams.py

>>>>>>Inputs:
* None

>>>>>>Functionalities:
* Mesh grid interval and overlap arrays

* Generate a list of parameters used by TDA inputs

>>>>>>Produce:
* 'data/session/params.json'
    e.g., [{"assetCode": "_130", "interval": 25, "overlap": 70},.........]

>>>>>>Run command from terminal:
* python3 mainParams.py data/session
    
"""

from __init__ import confUser
from tools import genIODic
from filters import FilterKnowledge
from json import dump
import sys
import os
from tools import progressPrinter
import time


confUser['DATADIR'] = sys.argv[1]
interval = confUser['interval']
overlap = confUser['overlap']

fn_params = os.path.join(confUser['DATADIR'], 'params.json')


n = .33
print('<<<{0:.2f}>>>'.format(1.*n))
sys.stdout.flush()
# Constract params.json
ioDic = genIODic(interval, overlap)


print('<<<{0:.2f}>>>'.format(2.*n))
sys.stdout.flush()
paramsForGenjson=[]
for kp, vp in ioDic.items():
    for kf in ['_107']: # , '_170', '_130']:
        epoch = { 'interval': int(vp[0]), 'overlap': int(vp[1]), 'assetCode': kf }
        paramsForGenjson.append(epoch)

with open(fn_params, 'w') as f:
    dump(paramsForGenjson, f)

print('<<<1>>>')
