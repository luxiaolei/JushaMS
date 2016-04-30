"""
/mainGenparams.py

when called from command line $JUSHAPATH/$: python mainClean.py AbsDatadir
This script do the following stuff:

"""
from __init__ import confUser
from tools import genIODic
from filters import FilterKnowledge
from json import dump
import sys
import os
from tools import progressPrinter

if __name__=='__main__':
	confUser['DATADIR'] = sys.argv[1]
	interval = confUser['interval']
	overlap = confUser['overlap']

	pgPrinter = progressPrinter(-.2, .2)
	pgPrinter.printStep
	sys.stdout.flush()

	ioDic = genIODic(interval, overlap)
	print('Construct iterval and overlap mesh grid done!')
	pgPrinter.printStep
	sys.stdout.flush()

	Filter = FilterKnowledge(confUser, v=False)
	filterDic = Filter.assetFilterDic(dim= 2 ,intresedCode= [107, 130, 170], genF= False)
	print('Construct filters Dictionary done!')
	pgPrinter.printStep
	sys.stdout.flush()

	paramsForGenjson=[]
	for kp, vp in ioDic.items():
		for kf, vf in filterDic.items():
			epoch = { 'interval': int(vp[0]), 'overlap': int(vp[1]), 'assetCode': kf }
			paramsForGenjson.append(epoch)
			pgPrinter.printStep
	print('Construct paramters for the next step is done!')
	pgPrinter.printStep
	sys.stdout.flush()

	paramsDicDir = os.path.join(confUser['DATADIR'], 'params.json')
	with open(paramsDicDir, 'w') as f:
		dump(paramsForGenjson, f)
	print('<<<1>>>')
	

