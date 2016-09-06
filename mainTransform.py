"""
mainTranform.py
>>>>>>Inputs:
* 'data/session/masterDf.csv'
* 'data/session/cleaned.csv'
>>>>>>Functionalities:
* Read 'data/session/masterDf.csv'
* Feature selection using low variance filter
* Feature selection using ANOVA-F test
* Minmax scale before SVM
* Model selection for SVM and Randomforest using RandomizedCrossValidation
* Generate Filters using distance to hyperplane
* Generate Silimarity vector by producting feature weights of descision tree with Anonva-F selected features (only for 107)
* Reproduce Cleaned.csv with rankings joint using both svm and randomforest
>>>>>>Produce:
* data/session/FilterDic.pkl 
    dict type with key '_107', '_170', '_130'. values are 1-D array
* data/session/_107_data.pkl
* data/session/_170_data.pkl
* data/session/_130_data.pkl
    values are 1-D array with length (n_samplesize * n_sample)/2 - n_samplesize
* data/session/Cleaned.csv
    with ranking columns, ['Rank_107', 'Rank_170', 'Rank_130'], joint to cleaned.csv
    
>>>>>>Run command from terminal:
* python3 mainTransform.py data/session
    
"""




import warnings
warnings.filterwarnings('ignore')

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist

import pickle as pkl
import numpy as np
import pandas as pd
import os
import sys

selectedCols = ['性别', '年龄', '婚姻', '学历', '从属行业', '理财抗风险等级', '客户层级', 
                '新老客户标记', '五级分类', '小微客户类型', '消费类资产产品', '纯消费性微贷标记', 
                '纯质押贷款标记', '非储投资偏好', '高金融资产标记', '购买大额他行理财标记', '大额消费标记', 
                '信用卡高还款标记', '信用卡高端消费标记', '优质行业标记', '高代发额客户标记', '潜在高端客户标记', 
                '客户贡献度', '客户活跃度', '客户渠道偏好', '客户金融资产偏好', '赎回理财次数', 
                '网银购买理财次数', '贵宾客户等级描述', '账户即时通签约标志', '活期余额', '存款余额月日均', 
                '存款余额年日均', '存款占比', '理财占比', '金融资产余额', '金融资产余额月日均', '金融资产余额年日均', 
                '持有定期存款标志', '手机银行签约标志', '持有定期存款标志', '手机银行签约标志', '三方存管签约标志', 
                '网银签约标志', '代发工资签约标志', '信用卡绑定还款签约标志', '按揭贷款标志', '当年购买理财标志', 
                '钱生钱签约标志', '资金归集签约标志', '乐收银签约标志', '开户年数', '金融资产余额_100', '金融资产余额_102']

#read dfXY
dataDir = sys.argv[1]

n = .2
print('<<<{0:.2f}>>>'.format(1.*n))
sys.stdout.flush()
#fn_cleand = os.path.join(dataDir, 'cleaned.csv')
fn_masterDf = os.path.join(dataDir, 'MasterDf.csv')
fn_filter = os.path.join(dataDir, 'FilterDic.pkl')


print('<<<{0:.2f}>>>'.format(2.*n))
sys.stdout.flush()
##select X based on selectedCols, select Y cols
dfXY = pd.read_csv(fn_masterDf, index_col='核心客户号')
dfX = dfXY[selectedCols]
dfYs = dfXY[['Y107', 'Y170', 'Y130']]


print('<<<{0:.2f}>>>'.format(3.*n))
sys.stdout.flush()
#ln transform balance features, minmax all afterwards
colsWithBalance = [col for col in dfX.columns if '金额' in col or '余额' in col]
for cb in colsWithBalance:
    dfX[cb] = dfX[cb].apply(np.log).apply(lambda x: x if x >=1 else 0)

scaler = MinMaxScaler()
X = scaler.fit_transform(dfX)


print('<<<{0:.2f}>>>'.format(4.*n))
sys.stdout.flush()
#Gen filters for each target

FilterDic = {}
for target in ['Y107', 'Y170', 'Y130']:
    SVCclf = svm.SVC(C=16)
    SVCclf.fit(X, dfYs[target])
    svmfilter = SVCclf.decision_function(X)
    SimilarityArr = pdist(X, metric='euclidean')
    name = '_'+target[1:]
    FilterDic[name] = svmfilter
    #generates Similarity for each asset
    fn_similarity = os.path.join(dataDir, name+'_data.npy')
    np.save(fn_similarity, SimilarityArr)

with open(fn_filter, 'wb') as f:
    pkl.dump(FilterDic, f)
print('<<<1>>>')
