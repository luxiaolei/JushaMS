"""
/mainTrans.py

when called from command line $JUSHAPATH/$: python mainTransform.py AbsDatadir
This script do the following stuff:

Generates filterDic.pkl
"""
import sys

from __init__ import confUser
from filters import FilterKnowledge





if __name__=='__main__':
	inputDir = sys.argv[1]
	confUser['DATADIR'] = inputDir
	print(confUser['DATADIR'])
	Filter = FilterKnowledge(confUser)
	Filter.assetFilterDic(dim= confUser['dim'] ,intresedCode= [107, 130, 170], genF= True)
	#print('<<<1>>>')


