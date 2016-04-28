"""
/mainTrans.py

when called from command line $JUSHAPATH/$: python mainClean.py AbsDatadir
This script do the following stuff:

"""
import sys

from __init__ import confUser
from preporcess import CleanerAnny





if __name__=='__main__':
	#inputDir = sys.argv[1]
	#confUser['DATADIR'] = inputDir
	print(confUser['DATADIR'])
	cleannerAnny = CleanerAnny(confUser, ['金融资产'])
	print(cleannerAnny.startCleaning().shape)


