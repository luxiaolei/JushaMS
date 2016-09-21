"""
mainMultiThread.py
>>>>>>>>>>Run command from terminal:
* python3 mainMultiThread.py $DATA_DIR
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import gc
import os
import time
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from os import path
from datetime import datetime
from multiprocessing import cpu_count

import pickle as pkl
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist
from sklearn import svm

from __init__ import *
from tools import genIODic, to_d3js_graph
import jushacore as mapper

#####################################
##          GLOBAL VAR             ##
#####################################

now = datetime.now()
print('================ Start at ' + str(now) + ' ================', flush = True)

ioPool = ThreadPoolExecutor(max_workers = 1)
computePool = ThreadPoolExecutor(max_workers = 4)

dataDir = sys.argv[1]
confUser['DATADIR'] = dataDir
interval = confUser['interval']
overlap = confUser['overlap']
metaJsonFile = path.join(dataDir, 'metadata.json')
metaJson = json.load(open(metaJsonFile, 'r', encoding = 'utf8'))

yCols = ['Y107', 'Y170', 'Y130']

#####################################
##          FUNCTIONS              ##
#####################################

def print_msg(msg):
    print('========= Elapsed time: {0:.2f} sec ========='.format((datetime.now() - now).total_seconds()), flush = True)
    print(msg, flush = True)

def save_metadata_json_file():
    global metaJsonFile
    global metaJson
    with open(metaJsonFile, 'w', encoding = 'utf8') as f:
        json.dump(metaJson, f, ensure_ascii = False, sort_keys = True, indent = 4)

def update_metadata():
    ioPool.submit(save_metadata_json_file)

def init_progress(key):
    global metaJson
    metaJson[key] = 0.0
    update_metadata()

def incr_progress(key, p):
    global metaJson
    metaJson[key] += p
    update_metadata()
    print_msg('<<<<<Progress[{0}]: {1:.2f}>>>>>'.format(key, metaJson[key]))

def progress_done(key):
    global metaJson
    metaJson[key] = 1
    update_metadata()
    print_msg('<<<<<Progress[' + key + ']: 1>>>>>')

def progress_failed(key):
    global metaJson
    metaJson[key] = -1
    update_metadata()
    print_msg('<<<<<Progress[' + key + ']: -1>>>>>')
    time.sleep(2)
    sys.exit(1)

def all_done():
    print_msg('ALL DONE!')

#####################################
##          CLEAN                  ##
#####################################

init_progress('cleaned')

rawdataDir = path.join(dataDir, 'rawdata')

def getDfs(kw):
    print_msg('I am looking for: ' + kw + ' in ' + rawdataDir)
    paths = []
    for dirpath, dirnames, filenames in os.walk(rawdataDir):
        for fn in filenames:
            if kw in fn and fn.endswith('.txt'):
                #fn has pattern: *keword*.txt*
                fnPath = path.join(dirpath, fn)
                paths.append(fnPath)
    # read_csv and catch errors
    assert len(paths) >= 1
    dfs = []
    for k, fn in enumerate(paths):
        try:
            df = pd.read_csv(fn, delimiter= '|', encoding='gbk', error_bad_lines = False)
            # for p=1, '\t' is also a delimiter, special case
            if df.shape[1] < 2:
                df = pd.read_csv(fn, delimiter='\t', encoding='gbk', error_bad_lines = False)
                if df.shape[1] < 2:
                    df = pd.read_csv(fn, delimiter=',', encoding='gbk', error_bad_lines = False)
            # 未知 can not be accepted by pandas
            df.replace(['未知', '其他', '?', ' '], np.nan, inplace = True)
            if k > 0:
                assert df.shape[1] == dfs[k - 1].shape[1]
            dfs.append(df)
        except Exception as e:
            print_msg('File: ' + fn + ', error in getting file porcess!')
    return dfs

def dfBuilder(kw, drop = True):
    dfs = getDfs(kw)
    dfFull = pd.concat(dfs, axis = 0)
    if drop:
        for uidName in ['核心客户号', 'khdm', '客户代码']:
            try:
                dfFull.drop_duplicates(subset = [uidname], inplace = True)
            except Exception:
                pass
    return dfFull

##################### 用户画像 ##########################

def load_user_image():
    try:
        dfuimageRaw = dfBuilder('客户画像').ix[:, [
                       '客户代码', '性别', '年龄', '婚姻', '学历', '从属行业', '理财抗风险等级', '客户层级', '新老客户标记',
                      '五级分类', '小微客户类型', '消费类资产产品', '纯消费性微贷标记', '纯质押贷款标记', '非储投资偏好',
                      '高金融资产标记', '购买大额他行理财标记', '大额消费标记', '信用卡高还款标记', '信用卡高端消费标记',
                      '优质行业标记', '高代发额客户标记', '潜在高端客户标记', '客户贡献度', '客户活跃度', '客户渠道偏好',
                      '客户金融资产偏好']]
        # select cols
        dfuimage = dfuimageRaw.copy()
        print_msg('Uimage Raw shape: ' + str(dfuimage.shape))

        # outlier
        dfuimage['年龄'] = dfuimage['年龄'].apply(lambda x: x if x < 120 else 120)

        # mapping
        UimageMapTable = pd.read_csv(maptableDir, header = None)
        UimageMapTable = UimageMapTable.ix[:, :1].values
        UimageMapTable = np.vstack((UimageMapTable,
            ['科学研究、技术服务和地质勘探业', 3],
            ['非储中贵金属偏好', 3],
            ['商铺按揭,住房按揭', 3],
            ['商铺按揭,消费型微贷,', 3],
            ['其他消费类,消费型微', 3]))
        dfuimage.rename(columns = { '客户代码': '核心客户号' }, inplace = True)
        for k,v in UimageMapTable:
            try:
                dfuimage.replace(k, v, inplace = True)
            except Exception as ex:
                print(ex, flush = True)
        print_msg('Uimage After clean shape: ' + str(dfuimage.shape))
    except Exception:
        progress_failed('cleaned')
    else:
        incr_progress('cleaned', 0.15)
    return (dfuimage, dfuimageRaw)

future_load_user_image = computePool.submit(load_user_image)

##################### 基本信息 ##########################

def load_user_info():
    try:
        dfuinfo = dfBuilder('基本信息')
        print_msg('Uinfo Raw shape: ' + str(dfuinfo.shape))
        # selectCols
        dfuinfo = dfuinfo.loc[:, [
                     '核心客户号', '首次开户日期', '贵宾客户等级描述', '账户即时通签约标志',  '活期余额', '存款余额月日均',
                     '存款余额年日均', '存款占比', '理财占比', '金融资产余额', '金融资产余额月日均', '金融资产余额年日均',
                     '持有定期存款标志', '手机银行签约标志', '三方存管签约标志', '网银签约标志', '代发工资签约标志',
                     '信用卡绑定还款签约标志', '按揭贷款标志',
                     '当年购买理财标志', '钱生钱签约标志',  '资金归集签约标志', '乐收银签约标志',
                     '近三个月柜面存款次数', '近三个月柜面存款金额', '近三个月柜面取款次数', '近三个月柜面取款金额',
                     '近三个月柜面转账次数', '近三个月柜面转账金额', '近三个月ATM存款次数', '近三个月ATM存款金额',
                     '近三个月ATM取款次数', '近三个月ATM取款金额', '近三个月网银转账次数', '近三个月网银转账金额',
                     '近三个月手机银行转账次数', '近三个月手机银行转账金额', '近三个月手机银行支付交易次数',
                     '近三个月手机银行支付交易金额', '近三个月手机银行缴费次数', '近三个月手机银行缴费金额',
                     '近三个月手机银行手机充值次数', '近三个月手机银行手机充值金额', '近三个月POS消费次数',
                     '近三个月POS消费金额', '近三个月跨行资金归集交易次数', '近三个月跨行资金归集交易金额',
                     '近三个月跨行通交易次数', '近三个月跨行通交易金额', '近三个月交易次数合计', '交易活跃度描述']]
        mappingTable = { '无效户': 0, '有效户': 1, '私人': 2, '银卡': 3, '金卡': 4, '钻卡': 5 }
        # transform
        dfuinfo['贵宾客户等级描述'] = dfuinfo['贵宾客户等级描述'].map(lambda x: mappingTable[x])
        dfuinfo['首次开户日期'] = pd.to_datetime(dfuinfo['首次开户日期'], infer_datetime_format = True)
        dfuinfo['开户年数'] = dfuinfo['首次开户日期'].map(lambda x: now.year - x.year )
        dfuinfo['交易活跃度描述'] = dfuinfo['交易活跃度描述'].apply(lambda x: 1 if x == '高活跃' else 0)
        #drop
        dfuinfo.drop(['首次开户日期'], axis = 1, inplace = True)
        print_msg('Uinfo after clean shape: ' + str(dfuinfo.shape))
    except Exception:
        progress_failed('cleaned')
    else:
        incr_progress('cleaned', 0.15)
    return dfuinfo

future_load_user_info = computePool.submit(load_user_info)

##################### 产品交易 ##########################

def load_trade():
    try:
        dftrade = dfBuilder('产品交易')
        print_msg('Trade Raw shape: ' + str(dftrade.shape))

        # rename index column
        dftrade.rename(columns = { '核心客户号': 'HXKHH' }, inplace = True)
        dftrade.rename(columns={ 'khdm': '核心客户号' }, inplace = True)

        #drop
        dftrade.drop(labels = ['统计日期', 'HXKHH', 'CUST_NAME'], axis = 1, inplace = True)
        dftrade.dropna(inplace = True)
        print_msg('Trade after clean shape: ' + str(dftrade.shape))
    except Exception:
        progress_failed('cleaned')
    else:
        incr_progress('cleaned', 0.15)
    return dftrade

future_load_trade = computePool.submit(load_trade)

##################### 金融资产 ##########################

def load_asset():
    try:
        dfasset = dfBuilder('金融资产', drop = False)
        print_msg('Asset Raw shape: ' + str(dfasset.shape))

        if not Consider0BalanceAsPositive:
            uid107 = dfasset.ix[(dfasset['金融资产代码'] == 107) & (dfasset['金融资产余额'] != 0), '核心客户号'].drop_duplicates().values
            uid170 = dfasset.ix[(dfasset['金融资产代码'] == 170) & (dfasset['金融资产余额'] != 0), '核心客户号'].drop_duplicates().values
            uid130 = dfasset.ix[(dfasset['金融资产代码'] == 130) & (dfasset['金融资产余额'] != 0), '核心客户号'].drop_duplicates().values
        else:
            uid107 = dfasset.ix[dfasset['金融资产代码'] == 107, '核心客户号'].drop_duplicates().values
            uid170 = dfasset.ix[dfasset['金融资产代码'] == 170, '核心客户号'].drop_duplicates().values
            uid130 = dfasset.ix[dfasset['金融资产代码'] == 130, '核心客户号'].drop_duplicates().values

        # select rows based on none-info leaking asset codes
        dfasset = dfasset.ix[(~dfasset['金融资产代码'].isin([107, 170, 130, 131, 1071, 1072, 104, 121, 150]) & (dfasset['金融资产余额'] != 0)), :]

        # flattern assets
        dfs = []
        for code, df in dfasset.groupby('金融资产代码'):
            df['金融资产余额_' + str(code)] = df['金融资产余额'].values
            df['金融资产余额季日均_' + str(code)] = df['金融资产余额季日均'].values
            newdf = df.ix[:, ['核心客户号', '金融资产余额_' + str(code), '金融资产余额季日均_' + str(code)]]
            # user can not hold one asset with multipul records
            newdf.drop_duplicates(subset = '核心客户号', inplace = True)
            newdf.set_index('核心客户号', drop = True, inplace = True)
            dfs.append(newdf)

        dfasset_dummy = dfs[0].join(dfs[1:])
        dfasset_dummy.fillna(0, inplace = True)
        print_msg('Asset after clean shape: ' + str(dfasset_dummy.shape))
    except Exception:
        progress_failed('cleaned')
    else:
        incr_progress('cleaned', 0.15)
    return (dfasset_dummy, uid107, uid170, uid130)

future_load_asset = computePool.submit(load_asset)

##################### JOIN DATA ##########################

(dfuimage, dfuimageRaw) = future_load_user_image.result()

dfuinfo = future_load_user_info.result()

dftrade = future_load_trade.result()

(dfasset_dummy, uid107, uid170, uid130) = future_load_asset.result()

dfuimage['Y107'] = dfuimage['核心客户号'].apply(lambda x: 1 if x in uid107 else 0)
dfuimage['Y170'] = dfuimage['核心客户号'].apply(lambda x: 1 if x in uid170 else 0)
dfuimage['Y130'] = dfuimage['核心客户号'].apply(lambda x: 1 if x in uid130 else 0)

dfuimage.set_index('核心客户号', drop = True, inplace = True)
dftrade.set_index('核心客户号', drop = True, inplace = True)
dfuinfo.set_index('核心客户号', drop = True, inplace = True)

dfXY = dfuimage.join([dftrade, dfuinfo, dfasset_dummy], how = 'inner')
dfXY.fillna(0, inplace = True)

# warning! dfXY.drop_duplicates(on all features).shape != (on uid only).shape
dfXY = dfXY.reset_index().drop_duplicates(subset = '核心客户号').set_index('核心客户号')
if dfXY.shape[0] == 0:
    raise Exception('No data at all!')

print_msg('Joint Master dataframe shape: ' + str(dfXY.shape))

def save_master_df(dfXY):
    try:
        dfXY.to_csv(path.join(dataDir, 'MasterDf.csv'))
    except Exception:
        progress_failed('cleaned')
    else:
        incr_progress('cleaned', 0.15)
ioPool.submit(save_master_df, dfXY)

##################### Generate cleaned.csv ##########################

def save_cleaned_csv(dfCleaned):
    try:
        dfCleaned.to_csv(path.join(dataDir, 'cleaned.csv'), index = False)
    except Exception:
        progress_failed('cleaned')
    else:
        progress_done('cleaned')

def generate_cleaned_csv():
    try:
        dforgs = dfBuilder('对应机构')
        dforgs.drop_duplicates(inplace = True)
        dforgs.set_index('khdm', drop = True, inplace = True)
        dforgs.rename(columns = { 'jgmc': '所属机构' }, inplace = True)

        global dfXY
        dfXYimage = dfXY[dfuimage.columns.values]
        print_msg('XYimageshape: ' + str(dfXYimage.shape))

        dfCleaned = dfXYimage.join(dforgs, how = 'left')
        print_msg('after join orgs, cleaned.csv shape: ' + str(dfCleaned.shape))

        # convert to raw uimage data
        dfuimageRaw.set_index('客户代码', drop = True, inplace = True)
        dfuimageRaw.drop_duplicates(inplace = True)
        dfCleaned = dfCleaned.ix[:, -4:].join(dfuimageRaw, how = 'left')
        print_msg('after join uimageRaw, cleaned.csv shape: ' + str(dfCleaned.shape))

        dfCleaned[yCols] = dfCleaned[yCols].astype(str)
        dfCleaned['核心客户号'] = dfCleaned.index.values
        dfCleaned.drop_duplicates(subset = ['核心客户号'], inplace = True)
        dfCleaned.rename(columns = { 'Y107': '主动负债', 'Y170': '保险', 'Y130': '基金' }, inplace = True)
        dfCleaned.replace(to_replace = { '主动负债': { "0": '未持有', "1": '持有' },
                                         '保险': { "0": '未持有', "1": '持有' },
                                         '基金': { "0": '未持有', "1": '持有' } }, inplace = True)

        print_msg('After drop dups UID, cleaned.csv shape: ' + str(dfCleaned.shape))
        ioPool.submit(save_cleaned_csv, dfCleaned)

        assert dfXY.shape[0] == dfCleaned.shape[0]
    except Exception:
        progress_failed('cleaned')
    else:
        incr_progress('cleaned', 0.15)
    gc.collect()

computePool.submit(generate_cleaned_csv)

#####################################
##          TRANSFORM              ##
#####################################

init_progress('transformed')

print_msg('Transforming data...')

# select X cols
dfX = dfXY[['性别', '年龄', '婚姻', '学历', '从属行业', '理财抗风险等级', '客户层级',
            '新老客户标记', '五级分类', '小微客户类型', '消费类资产产品', '纯消费性微贷标记',
            '纯质押贷款标记', '非储投资偏好', '高金融资产标记', '购买大额他行理财标记', '大额消费标记',
            '信用卡高还款标记', '信用卡高端消费标记', '优质行业标记', '高代发额客户标记', '潜在高端客户标记',
            '客户贡献度', '客户活跃度', '客户渠道偏好', '客户金融资产偏好', '赎回理财次数',
            '网银购买理财次数', '贵宾客户等级描述', '账户即时通签约标志', '活期余额', '存款余额月日均',
            '存款余额年日均', '存款占比', '理财占比', '金融资产余额', '金融资产余额月日均', '金融资产余额年日均',
            '持有定期存款标志', '手机银行签约标志', '持有定期存款标志', '手机银行签约标志', '三方存管签约标志',
            '网银签约标志', '代发工资签约标志', '信用卡绑定还款签约标志', '按揭贷款标志', '当年购买理财标志',
            '钱生钱签约标志', '资金归集签约标志', '乐收银签约标志', '开户年数', '金融资产余额_100', '金融资产余额_102']]
# select Y cols
dfYs = dfXY[yCols]

# ln transform balance features
for c in [col for col in dfX.columns if '金额' in col or '余额' in col]:
    dfX[c] = dfX[c].apply(np.log).apply(lambda x: x if x >= 1 else 0)
# minmax all
scaler = MinMaxScaler()

def generate_filter(target, X):
    name = '_' + target[1:]
    print_msg('Generating Filter of ' + target + '...')
    SVCclf = svm.SVC(C = 16)
    SVCclf.fit(X, dfYs[target])
    svmfilter = SVCclf.decision_function(X)
    print_msg('Filter of ' + target + ' generated.')
    return (name, svmfilter)

def save_filters(filters):
    try:
        with open(path.join(dataDir, 'FilterDic.pkl'), 'wb') as f:
            pkl.dump(filters, f)
    except Exception:
        progress_failed('transformed')
    else:
        progress_done('transformed')

def calculte_distance_matrix(X):
    print_msg('Calculating distance matrix...')
    print_msg('<<<<<Progress[results]: 0.0>>>>>')
    return pdist(X, metric = 'euclidean')
    # return pairwise_distances(X, metric = 'euclidean', n_jobs = 16)

try:
    X = scaler.fit_transform(dfX)
except Exception:
    progress_failed('transformed')
else:
    incr_progress('transformed', 0.3)
    future_calculte_distance_matrix = computePool.submit(calculte_distance_matrix, X)

print_msg('Generating filters...')
step = 0.6 / len(yCols)
filters = {}
future_to_target = { computePool.submit(generate_filter, target, X): target for target in yCols }
for future in futures.as_completed(future_to_target):
    target = future_to_target[future]
    try:
        (name, svmfilter) = future.result()
    except Exception as ex:
        print('Filter %r generated an exception: %s.' % (target, ex))
        progress_failed('transformed')
    else:
        incr_progress('transformed', step)
        filters[name] = svmfilter
ioPool.submit(save_filters, filters)

gc.collect()

#####################################
##          PARAMS                 ##
#####################################

init_progress('params')

print_msg('Calculating parameters...')

jsonNameTails = list(map(lambda x: '_' + x[1:], yCols))

def save_params_json(params):
    try:
        with open(path.join(dataDir, 'params.json'), 'w', encoding = 'utf8') as f:
            json.dump(params, f, ensure_ascii = False, sort_keys = True, indent = 4)
    except Exception:
        progress_failed('params')
    else:
        progress_done('params')

def generate_params():
    params = []
    ioDic = genIODic(interval, overlap)
    for kp, vp in ioDic.items():
        for kf in jsonNameTails:
            params.append({ 'interval': int(vp[0]), 'overlap': int(vp[1]), 'assetCode': kf })
    incr_progress('params', 0.5)
    ioPool.submit(save_params_json, params)
    return params

params = generate_params()

gc.collect()

#####################################
##          RESULTS                ##
#####################################

metaJson['results'] = []

def param_to_file_name(interval, overlap, assetCode):
    return 'i' + str(interval) + 'o' + str(overlap) + assetCode

def core_wrapper(resultsDir, data, interval, overlap, assetCode, file_name):
    filter = filters[assetCode]
    print_msg('Data shape: ' + str(data.shape) + ', Filter shape: ' + str(filter.shape))
    print_msg('Calculating topology graph of ' + file_name + '...')
    try:
        cover = mapper.cover.cube_cover_primitive(interval, overlap)
        mapper_output = mapper.jushacore(data, filter, cover = cover, cutoff = None,
                                         cluster = mapper.single_linkage(),
                                         metricpar = { 'metric': 'euclidean' },
                                         verbose = False)
        mapper.scale_graph(mapper_output, filter, cover = cover, weighting = 'inverse',
                           exponent = 1, verbose = False)
    except Exception:
        return -1
    else:
        if mapper_output.stopFlag:
            print_msg(file_name + ' Stopped! Too many nodes or too long time')
            return -1
        else:
            print_msg(file_name + ' Successed!')
            print_msg('type check: ' + str(type(mapper_output)))
            to_d3js_graph(mapper_output, file_name, resultsDir, True)
            print_msg('Core ran finished! with: ' + file_name)
            return 1

try:
    dist_matrix = future_calculte_distance_matrix.result()
except Except:
    progress_failed('results')
else:
    print_msg('<<<<<Progress[results]: 0.3>>>>>')

gc.collect()

print_msg('Calculating topology graph...')
resultsDir = path.join(dataDir, 'results')
p = 0.3
step = 0.7 / len(params)
for param in params:
    (i, o, a) = (param['interval'], param['overlap'], param['assetCode'])
    file_name = param_to_file_name(i, o, a)
    try:
        status = core_wrapper(resultsDir, dist_matrix, i, o, a, file_name)
    except Exception as ex:
        status = -1
        print('Result %r generated an exception: %s' % (file_name, ex))
    print_msg('!!!!!!!! ' + file_name + ': ' + status + ' !!!!!!!!')
    metaJson['results'].append({ file_name + '.json': status })
    update_metadata()
    p += step
    print_msg('<<<<<Progress[results]: {0:.2f}>>>>>'.format(p))

print_msg('<<<<<Progress[results]: 1!>>>>>')
all_done()

computePool.shutdown()
ioPool.shutdown()
gc.collect()
