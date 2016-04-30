"""
/mainClean.py

when called from command line $JUSHAPATH/$: python mainClean.py AbsDatadir
This script do the following stuff:

1.
"""
import sys

from __init__ import confUser
from preporcess import Cleaner, CleanerAnny
from tools import progressPrinter
import pandas as pd
import os

#inputDir = sys.argv[1]
#confUser['DATADIR'] = sys.argv[1]





if __name__=='__main__':

	print(confUser['DATADIR'])

	confUser['DATADIR'] = sys.argv[1]

	maptableDir = 'config/mappingTable.csv'
	usefulCols = ['性别', '年龄', '婚姻', '学历', '从属行业',
	'理财抗风险等级', '客户层级', '新老客户标记', '五级分类', '小微客户类型', '消费类资产产品', '纯消费性微贷标记',
	'纯质押贷款标记', '非储投资偏好', '高金融资产标记', '购买大额他行理财标记', '大额消费标记', '信用卡高还款标记',
	'信用卡高端消费标记', '优质行业标记', '高代发额客户标记','潜在高端客户标记', '客户贡献度',
	'客户活跃度', '客户渠道偏好', '客户金融资产偏好']

	
	cleaner = Cleaner(confUser)
	
	cleaner.startCleaning(maptableDir, usefulCols)

	cleannerAnny = CleanerAnny(confUser, ['金融资产'], v=False)
	cleaner.saveNlog()

	dfYall = cleannerAnny.startCleaning(intresedCode= [107, 130, 170])
	#dfYall.rename(columns={'核心客户号':'jgmc'},inplace=True)

	dfYallcsv = os.path.join(confUser['DATADIR'], 'dfYall.csv')
	dfYall.to_csv(dfYallcsv, index=False)
 	#cleaner.cleanedDf['核心客户号'] = cleaner.cleanedDf.index
	cleaner.cleanedDf['核心客户号'] = cleaner.cleanedDf.index.values
	cleanedDf = pd.merge(cleaner.cleanedDf, dfYall, on='核心客户号')
	fnc = os.path.join(confUser['DATADIR'], 'cleaned.csv')
	cleanedDf.to_csv(fnc, index=False)
	

