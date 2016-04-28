"""
/__main__.py

#to run package as a cleanner with rawdata dir name specified
bash$: python3 jushaMinsheng -c datapath

#to generate jsons
bash$: python3 jushaMinsheng -r  
"""
from _jusha import confUser
from _jusha.preporcess import Cleaner, CleanerAnny
from _jusha.filters import FilterKnowledge
from _jusha.wrapper import coreWrapper
import _jusha.jushacore as mapper


print(confUser)

t0 = time.time()

#clean data and gen cleaned.csv transformed.csv
cleaner = Cleaner(confUser)
cleaner.startCleaning(maptableDir, usefulCols)
cleaner.saveNlog()
if cleaner.error:
    print('error msg: ',cleaner.error, cleaner.errorTrace)
else:
    print('Progress Cleanning and Transformation Done!')

#Construct filters
fileNameKeyword = ['金融资产']
cleannerAnny = CleanerAnny(confUser, fileNameKeyword)
KnowlgeF = FilterKnowledge(confUser)
print('Progress filter construction done!')

#Run jushaCore in multithreading fashion
intresedCode= [107, 130, 170]
coreWrapper = coreWrapper(confUser)
coreWrapper.assetStartWrapping(KnowlgeF, dim=2, intresedCode= intresedCode)

t1 = time.time()
print('for 11596,26 data, Dimension {0}, total time cost {1}').format(dim, t1-t0)


"""


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('config/config.ini')
    confUser = config['USER']
    
    if sys.args[1] == '-c':
        #start cleanning process, generates csvs
        confUser['DATADIR'] = 'data/'+ sys.arg[1]
        maptableDir = 'config/mappingTable.csv'
        usefulCols = ['性别', '年龄', '婚姻', '学历', '从属行业',
        '理财抗风险等级', '客户层级', '新老客户标记', '五级分类', '小微客户类型', '消费类资产产品', '纯消费性微贷标记',
        '纯质押贷款标记', '非储投资偏好', '高金融资产标记', '购买大额他行理财标记', '大额消费标记', '信用卡高还款标记',
        '信用卡高端消费标记', '优质行业标记', '高代发额客户标记','潜在高端客户标记', '客户贡献度',
        '客户活跃度', '客户渠道偏好', '客户金融资产偏好']
        cleaner = Cleaner(confUser)
        cleaner.startCleaning(maptableDir, usefulCols)
        cleaner.saveNlog()
    elif sys.arg[1] == '-r':
        #start jusharcore process, generates jsons
"""
        


