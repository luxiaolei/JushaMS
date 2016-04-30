# -*- coding: utf-8 -*-
#from tools import posMinDis



#import configparser
import numpy as np

"""
from preporcess import Cleaner, CleanerAnny
from filters import FilterKnowledge
from wrapper import coreWrapper
"""

confUser = {
	'DATADIR' : "data/tempdir",
	'maptableDir' : 'config/mappingTable.csv',
	"NANVALUE" : '?,未知',  #used by pd.read_csv(.., nanvalue= nanvalue), but not working porperly
	'EXTENSION' : '.txt',
	'DELI' : '|',
	'KEYWORD' : '客户画像,对应机构', #used by in transformation proprecess
	'ENCODING' : 'gb18030', #? redundent
    "interval" : np.arange(10, 30, 2),
    "overlap" : np.arange(75, 95, 3)
}

"""
config = configparser.ConfigParser()
config.read('config/config.ini')
confUser = config['USER']
confUser['DATADIR'] = 'data/tempdir'# 'data/'+ sys.arg[1]
"""

maptableDir = 'config/mappingTable.csv'

usefulCols = ['性别', '年龄', '婚姻', '学历', '从属行业',
'理财抗风险等级', '客户层级', '新老客户标记', '五级分类', '小微客户类型', '消费类资产产品', '纯消费性微贷标记',
'纯质押贷款标记', '非储投资偏好', '高金融资产标记', '购买大额他行理财标记', '大额消费标记', '信用卡高还款标记',
'信用卡高端消费标记', '优质行业标记', '高代发额客户标记','潜在高端客户标记', '客户贡献度',
'客户活跃度', '客户渠道偏好', '客户金融资产偏好']

