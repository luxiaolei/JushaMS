"""
mainClean.py
>>>>>>Functionalities:
* Load data from 'data/session/' directory
    file name contains '客户画像', '金融资产', '产品交易, '基本信息'
    
* Clean data
    Drop nan values
    map values according to maptable
    Outliers removal
    
* Transform data
    Drop columns and rows which are justified as overfitting leakers. (neglect this process when in production)
    Transform 金融资产 table by dummy codes
    Drop duplicates
* Inner Join dataframes
    Set '核心客户号' as indexs
    Inner join dataframes with last three columns as 'Y107','Y170','Y130'
* Generates Dataframe for visual
    Parse '对应机构' file and join to '客户画像' table.
    Join 'Y107','Y170','Y130' to '客户画像' table
* Print redirected to 'data/session/cleanLog.txt'
>>>>>>Produce:
* 'data/session/cleaned.csv' -Used for Qiangbao Visual
* 'data/session/MasterDf.csv' 
* 'data/session/CleanLog.txt'
>>>>>>Run command from terminal:
* python3 mainClean.py data/session
    
"""


from preporcess import CleanerAnny
from __init__ import *
import pandas as pd
import sys
import os

dataDir = sys.argv[1]

fn_log = os.path.join(dataDir, 'CleanLog.txt')
fn_cleand = os.path.join(dataDir, 'cleaned.csv')
fn_masterDf = os.path.join(dataDir, 'MasterDf.csv')
confUser['DATADIR'] = dataDir
datain = os.path.join(dataDir, 'rawdata')

#sys.stdout = open(fn_log, 'w')

def getDfs(kw):
    """
    extract filepath which statisfy keyword&ext conditions
    return list of df
    *args 
    """
    print('I am looking for:', kw, ' in ', datain)
    paths = []
    for dirpath, dirnames, filenames in os.walk(datain):
        for fn in filenames:
            if kw in fn and fn.endswith('.txt'):
                #fn has pattern: *keword*.txt*
                fnPath = os.path.join(dirpath, fn)
                paths.append(fnPath)

    #read_csv and catch errors
    assert len(paths)>= 1
    dfs = []
    for k, fn in enumerate(paths):
        try:
            df = pd.read_csv(fn, delimiter= '|', encoding='gbk', 
                              error_bad_lines= False)
            #未知 can not be accepted by pandas
            df.replace(['未知', '其他', '?', ' '], np.nan, inplace=True)

            #for p=1, '\t' is also a delimiter, special case
            if df.shape[1] < 2:
                df = pd.read_csv(fn, delimiter='\t', encoding='gbk', 
                                  error_bad_lines= False)

            if k> 0: assert df.shape[1]== dfs[k-1].shape[1]
            dfs.append(df)

        except Exception as e:
            msg = 'file: '+ fn+ ', error in getting file porcess!'
            print(msg)

    return dfs 

def dfBuilder(kw, drop=True):
    #cleaner = CleanerAnny(confUser, kw)
    dfs = getDfs(kw)
    dfFull = pd.concat(dfs, axis =0)
    if drop:
        for uidName in ['核心客户号', 'khdm', '客户代码']:
            try:
                dfFull.drop_duplicates(subset=[uidname], inplace=True)    
            except Exception:
                pass
    return dfFull

#some index are string, need to int it
def intIndex(x):
    try:
        return int(x)
    except Exception as e:
        return np.nan


    
############################################################################
uimageCols = ['客户代码', '性别', '年龄', '婚姻', '学历', '从属行业', '理财抗风险等级', '客户层级', '新老客户标记',
              '五级分类', '小微客户类型', '消费类资产产品', '纯消费性微贷标记', '纯质押贷款标记', '非储投资偏好', 
              '高金融资产标记', '购买大额他行理财标记', '大额消费标记', '信用卡高还款标记', '信用卡高端消费标记', 
              '优质行业标记', '高代发额客户标记', '潜在高端客户标记', '客户贡献度', '客户活跃度', '客户渠道偏好', '客户金融资产偏好']

uinfoCols = ['核心客户号', '首次开户日期', '贵宾客户等级描述', '账户即时通签约标志',  '活期余额', '存款余额月日均', '存款余额年日均',
         '存款占比', '理财占比', '金融资产余额', '金融资产余额月日均', '金融资产余额年日均',
         '持有定期存款标志', '手机银行签约标志', '三方存管签约标志', '网银签约标志', '代发工资签约标志', 
       '信用卡绑定还款签约标志', '按揭贷款标志',
       '当年购买理财标志', '钱生钱签约标志',  '资金归集签约标志', '乐收银签约标志',
       '近三个月柜面存款次数', '近三个月柜面存款金额', '近三个月柜面取款次数', '近三个月柜面取款金额', '近三个月柜面转账次数',
       '近三个月柜面转账金额', '近三个月ATM存款次数', '近三个月ATM存款金额', '近三个月ATM取款次数',
       '近三个月ATM取款金额', '近三个月网银转账次数', '近三个月网银转账金额', '近三个月手机银行转账次数',
       '近三个月手机银行转账金额', '近三个月手机银行支付交易次数', '近三个月手机银行支付交易金额', '近三个月手机银行缴费次数',
       '近三个月手机银行缴费金额', '近三个月手机银行手机充值次数', '近三个月手机银行手机充值金额', '近三个月POS消费次数',
       '近三个月POS消费金额', '近三个月跨行资金归集交易次数', '近三个月跨行资金归集交易金额', '近三个月跨行通交易次数',
       '近三个月跨行通交易金额', '近三个月交易次数合计', '交易活跃度描述']
mappingTable = {
    '无效户': 0,
    '有效户': 1,
    '私人': 2,
    '银卡': 3,
    '金卡': 4,
    '钻卡': 5
}
UimageMapTable = pd.read_csv(maptableDir, header=None)
UimageMapTable = UimageMapTable.ix[:, :1].values
UimageMapTable = np.vstack((UimageMapTable, 
    ['科学研究、技术服务和地质勘探业', 3],
    ['非储中贵金属偏好', 3],
    ['商铺按揭,住房按揭', 3],
    ['商铺按揭,消费型微贷,', 3],
    ['其他消费类,消费型微', 3]))
targetLeakCodes = [107, 170, 130, 131, 1071, 1072, 104, 121, 150]


############################################################################
dfuimageRaw = dfBuilder('客户画像', drop=True).ix[:, uimageCols]

#select cols
dfuimage = dfuimageRaw.copy()
print('Uimage Raw shape:', dfuimage.shape)
n = 0.1428
print('<<<{0:.2f}>>>'.format(1.*n))
sys.stdout.flush()

#outlier
dfuimage['年龄'] = dfuimage['年龄'].apply(lambda x: x if x < 120 else 120 )

#replace nan
dfuimage.replace(['未知', '其他', '?', ' '], np.nan, inplace=True)

#mapping
dfuimage.rename(columns={'客户代码': '核心客户号'}, inplace=True)
for k,v in UimageMapTable:
    try:
        dfuimage.replace(k,v, inplace=True)
    except Exception as e:
        print(e)
print('Uimage After clean shape: ', dfuimage.shape)
print('<<<{0:.2f}>>>'.format(2.*n))
sys.stdout.flush()
############################################################################
dfuinfo = dfBuilder('基本信息', drop=True)
print('Uinfo Raw shape:', dfuinfo.shape)

#slectCols
dfuinfo = dfuinfo.loc[:, uinfoCols]

#transform 
dfuinfo['贵宾客户等级描述'] = dfuinfo['贵宾客户等级描述'].map(lambda x: mappingTable[x])
dfuinfo['首次开户日期'] = pd.to_datetime(dfuinfo['首次开户日期'], infer_datetime_format=True)
dfuinfo['开户年数'] = dfuinfo['首次开户日期'].map(lambda x: 2016- x.year )
dfuinfo['交易活跃度描述'] = dfuinfo['交易活跃度描述'].apply(lambda x: 1 if x=='高活跃' else 0) 

#drop
dfuinfo.drop(['首次开户日期'],axis=1 , inplace=True)
print('Uinfo after clean shape:', dfuinfo.shape)
print('<<<{0:.2f}>>>'.format(3.*n))
sys.stdout.flush()
############################################################################
dftrade = dfBuilder('产品交易', drop=True)
print('Trade Raw shape:', dftrade.shape)

#index type convert
dftrade['核心客户号'] = dftrade['核心客户号'].apply(intIndex)

#drop
dftrade.drop(labels=['统计日期', 'khdm', 'CUST_NAME'], axis=1, inplace=True)
dftrade.dropna(inplace=True)
print('Trade after clean shape:', dftrade.shape)
print('<<<{0:.2f}>>>'.format(4.*n))
sys.stdout.flush()
############################################################################
dfasset = dfBuilder('金融资产', drop=False)
print('Asset Raw shape:', dfasset.shape)

if not Consider0BalanceAsPositive:
    uid107 = dfasset.ix[(dfasset['金融资产代码']==107)&(dfasset['金融资产余额']!=0), '核心客户号'].drop_duplicates().values
    uid170 = dfasset.ix[(dfasset['金融资产代码']==170)&(dfasset['金融资产余额']!=0), '核心客户号'].drop_duplicates().values
    uid130 = dfasset.ix[(dfasset['金融资产代码']==130)&(dfasset['金融资产余额']!=0), '核心客户号'].drop_duplicates().values
else:
    uid107 = dfasset.ix[dfasset['金融资产代码']==107, '核心客户号'].drop_duplicates().values
    uid170 = dfasset.ix[dfasset['金融资产代码']==170, '核心客户号'].drop_duplicates().values
    uid130 = dfasset.ix[dfasset['金融资产代码']==130, '核心客户号'].drop_duplicates().values


#select rows based on none-info leaking asset codes
dfasset = dfasset.ix[(~dfasset['金融资产代码'].isin(targetLeakCodes)&(dfasset['金融资产余额']!=0)), :]
#dfasset.drop_duplicates(inplace=True)

#flattern assets
uidGrouped = dfasset.groupby('金融资产代码')
dfs = []
for code, df in uidGrouped:
    df['金融资产余额_'+str(code)] = df['金融资产余额'].values
    df['金融资产余额季日均_'+str(code)] = df['金融资产余额季日均'].values
    newdf = df.ix[:, ['核心客户号', '金融资产余额_'+str(code), '金融资产余额季日均_'+str(code)]]
    newdf.drop_duplicates(subset='核心客户号', inplace=True) #user can not hold one asset with multipul records
   
    newdf.set_index('核心客户号', drop=True, inplace=True)
    dfs.append(newdf)

dfasset_dummy = dfs[0].join(dfs[1:])
dfasset_dummy.fillna(0, inplace=True)
print('Asset after clean shape:', dfasset_dummy.shape)
print('<<<{0:.2f}>>>'.format(5.*n))
sys.stdout.flush()
############################################################################
dfuimage['Y107'] = dfuimage['核心客户号'].apply(lambda x: 1 if x in uid107 else 0)
dfuimage['Y170'] = dfuimage['核心客户号'].apply(lambda x: 1 if x in uid170 else 0)
dfuimage['Y130'] = dfuimage['核心客户号'].apply(lambda x: 1 if x in uid130 else 0)

dfuimage.set_index('核心客户号', drop= True, inplace= True)
dftrade.set_index('核心客户号', drop= True, inplace= True)
dfuinfo.set_index('核心客户号', drop= True, inplace= True)

dfXY = dfuimage.join([dftrade, dfuinfo, dfasset_dummy], how='inner')
dfXY.fillna(0, inplace=True)

#warning! dfXY.drop_duplicates(on all features).shape != (on uid only).shape
dfXY = dfXY.reset_index().drop_duplicates(subset='核心客户号').set_index('核心客户号')
print('Joint Master dataframe shape:', dfXY.shape)
print('>>>>>'*20)

dfXY.to_csv(fn_masterDf)
print('<<<{0:.2f}>>>'.format(6.*n))
sys.stdout.flush()
############################################################################
##Generates Cleaned.csv

dforgs = dfBuilder('对应机构', drop=True)
dforgs.drop_duplicates(inplace=True)
dforgs.set_index('khdm', drop=True, inplace=True)
dforgs.rename(columns={'jgmc': '所属机构'}, inplace=True)

dfXYimage = dfXY[dfuimage.columns.values]
print('XYimageshape:', dfXYimage.shape)

dfCleaned = dfXYimage.join(dforgs, how='left')
print('after join orgs, cleaned.csv shape:', dfCleaned.shape)

#convert to raw uimage data
dfuimageRaw.set_index('客户代码', drop=True, inplace =True)
dfuimageRaw.drop_duplicates(inplace=True)
dfCleaned = dfCleaned.ix[:, -4:].join(dfuimageRaw, how='left')
print('after join uimageRaw, cleaned.csv shape:', dfCleaned.shape)

dfCleaned[['Y107', 'Y170', 'Y130']] = dfCleaned[['Y107', 'Y170', 'Y130']].astype(str)
dfCleaned['核心客户号'] = dfCleaned.index.values
dfCleaned.drop_duplicates(subset=['核心客户号'], inplace=True)
dfCleaned.rename(columns={'Y107':'主动负债', 'Y170': '保险', 'Y130': '基金'}, inplace=True)
dfCleaned.replace(to_replace={'主动负债': {"0": '未持有', "1": '持有'},
                            '保险': {"0": '未持有', "1": '持有'},
                            '基金': {"0": '未持有', "1": '持有'}}, inplace=True)

print('After drop dups UID, cleaned.csv shape:', dfCleaned.shape)
dfCleaned.to_csv(fn_cleand, index=False)
print('Final, cleaned.csv shape:', dfCleaned.shape)

assert dfXY.shape[0] == dfCleaned.shape[0]
print('<<<1>>>')
