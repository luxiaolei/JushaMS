"""
/mainGenparams.py

when called from command line $JUSHAPATH/$: python mainClean.py AbsDatadir
This script do the following stuff:

"""
from __init__ import confUser
from tools import genIODic
from filters import FilterKnowledge
from json import dump



if __name__=='__main__':
	confUser['DATADIR'] = sys.args[1]
	interval = confUser['interval']
	overlap = confUser['overlap']

	ioDic = genIODic(interval, overlap)
	print('Construct iterval and overlap mesh grid done!')

	Filter = FilterKnowledge(confUser)
	filterDic = Filter.assetFilterDic(dim= 2 ,intresedCode= [107, 130, 170], genF= True)
	print('Construct filters Dictionary done!')

	paramsForGenjson = []
	for kp, vp in paramDic.items():
     	for kf, vf in filterDic.items():
     		epoch = (vp[0], vp[1], kf)
     		paramsForGenjson.append(epoch)
 	print('Construct paramters for the next step is done!')

	paramsDicDir = os.path.join(confUser['DATADIR'], 'params.json')
	with open(paramsDicDir, 'wb') as f:
		dump(paramsForGenjson, f)
