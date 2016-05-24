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

from sklearn import svm, feature_selection, cross_validation
from sklearn.feature_selection import VarianceThreshold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist
from scipy.stats import randint as sp_randint
from operator import itemgetter

import pickle as pkl
import numpy as np
import pandas as pd
import os
import sys





def ihs(x, theta= 22.3): #theta=22.3 gives convergance to 1 at x goes 10**8
    return np.log(theta*x +((theta*x)**2 + 1)**.5)/theta


def FeatureSelection_Pipline(dfX, dfY, estimator, thlhdv = 0.98, scale=False):
    """
    Do low variance filter,
    Do Minmax scale if set to True
    Do Anova-F test to select the best percentile
    
    Arguments:
    dfX - raw data, (n_sample * n_features)
    dfY - raw data, (n_sample * 1)
    estimator - Classifier instance
    scale - True for minmaxscale as a pipline
    
    Reutrns:
    dfX - with best columns
    """
    #Low variance filter
    sel = VarianceThreshold(threshold=(thlhdv * (1 - thlhdv)))
    sel.fit(dfX.values)
    X = sel.transform(dfX.values)
    X_sel_cols = dfX.columns[sel.get_support()]
    
    #Minmax Scale and Anonva
    transform = feature_selection.SelectPercentile(feature_selection.f_classif)
    if scale:
        minMaxScaler = MinMaxScaler()
        clf = Pipeline([('MinmaxScaler', minMaxScaler), ('anova', transform), ('estimator', estimator)])
    else:
        clf = Pipeline([('anova', transform), ('estimator', estimator)])

    score_means = list()
    score_stds = list()
    score_maxs = list()
    percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

    for percentile in percentiles:
        clf.set_params(anova__percentile=percentile)
        # Compute cross-validation F1 score using all CPUs
        this_scores = cross_validation.cross_val_score(clf, X, dfY.values, cv=10, scoring='f1')
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())
        score_maxs.append(this_scores.max())

    bestPercentile = percentiles[np.argmax(score_means)]
    SelectF = feature_selection.SelectPercentile(feature_selection.f_classif, percentile=bestPercentile)
    SelectF.fit(X, dfY.values)
    selectedMask = SelectF.get_support()
    selectedCols = X_sel_cols[selectedMask]
    return dfX[selectedCols]

def ModelSelection(X, Y, clf, param_dist, n_iter_search = 300):
    """
    Do randomzied model selection
    
    Arguments:
    X - Dataframe or np.array
    Y - target
    clf - estimator instance
    param_dist - dict of list of parameters for clf
    n_iter_search - number of search times
    
    Returns:
    clf - with best params, fitted with X,Y
    f1_meanScore - mean f1 score associated with the best params
    """
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, scoring='f1')
    random_search.fit(X, Y)
    params, mean, scores = sorted(random_search.grid_scores_, key=itemgetter(1), reverse=True)[0]
    clf.set_params(**params)
    clf.fit(X, Y)
    return clf, mean


    
def wrapper(dfX, dfY):
    ###############################################################################
    ## Feature selection
    estimator = svm.SVC(C=500.0)
    dfX_selected = FeatureSelection_Pipline(dfX, dfY, estimator, thlhdv = 0.98, scale=True)
    n_cols = dfX_selected.shape[1]

    ###############################################################################
    ## SVC model selection. Generates filter
    SVCclf = svm.SVC()
    # specify parameters and distributions to sample from
    SVCparam_dist = {'C': np.concatenate((np.arange(.1, 5, .2), np.arange(5, 20, 2), np.arange(20, 100, 20))),
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'degree': [2, 3, 4],
                  'gamma': np.arange(1/(n_cols*3), 1, 1/(n_cols*3.5)),
                  'class_weight': ['balanced', None]}
    
    SVCclf, SVCmeanF1 = ModelSelection(dfX_selected, dfY, SVCclf, SVCparam_dist, n_iter_search = 100)
    svmfilter = SVCclf.decision_function(dfX_selected)     

    ###############################################################################
    ##After Anonva-SVM feature selection, Select the best model for RF
    ##Then, generates Similarity Array and Ranking
    RFclf = RandomForestClassifier()

    # specify parameters and distributions to sample from
    RFparam_dist = {'n_estimators': np.concatenate((np.arange(20, 200, 30), np.arange(200, 1000, 150))),
                  "max_depth": np.arange(1, n_cols, 1),
                  "max_features": sp_randint(1, n_cols),
                  "min_samples_split": sp_randint(1, dfX_selected.shape[0]),
                  "min_samples_leaf": sp_randint(1, dfX_selected.shape[0]),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "class_weight": ['balanced', 'balanced_subsample', None]}
    
    RFclf, RFmeanF1 = ModelSelection(dfX_selected, dfY, RFclf, RFparam_dist, n_iter_search = 100)

    #use the feature_importances to weight X, and then calculate Similarities using euclidean metric
    dfX_selected_weighted = dfX_selected * RFclf.feature_importances_
    SimilarityArr = pdist(dfX_selected_weighted.values, metric='euclidean')

    #rank the users by predict_proba method
    proba0 = RFclf.predict_proba(dfX_selected)[:, 0]
    ranksArr = proba0.argsort()
    return svmfilter, SimilarityArr, ranksArr


if __name__=='__main__':
    # Perpare filenames
    dataDir = sys.argv[1]

    
    fn_cleand = os.path.join(dataDir, 'cleaned.csv')
    fn_masterDf = os.path.join(dataDir, 'MasterDf.csv')
    fn_filter = os.path.join(dataDir, 'FilterDic.pkl')
    

    # Get X, Ys ready
    dfXY = pd.read_csv(fn_masterDf, index_col='核心客户号')
    dfX = dfXY.drop(['Y107', 'Y170', 'Y130', '当年购买理财标志', '理财交易次数', '购买理财次数', '手机银行购买理财次数'], axis=1)
    dfYs = dfXY[['Y107', 'Y170', 'Y130']]

    # IHS transform those 余额，金额 columns
    colsWithBalance = [col for col in dfX.columns if '金额' in col or '余额' in col]
    for col in colsWithBalance:
        dfX[col] = dfX[col].apply(ihs)
    
    # Generates results
    FilterDic = {}
    SimilarityDic = {}
    Cleandf = pd.read_csv(fn_cleand)
    for target in ['Y107']:#, 'Y170', 'Y130']:
        svmfilter, SimilarityArr, ranksArr = wrapper(dfX, dfYs[target])
        name = '_'+target[1:]
        FilterDic[name] = svmfilter
        Cleandf['Rank'+name] = ranksArr
        
        #generates Similarity for each asset
        fn_similarity = os.path.join(dataDir, name+'_data.pkl')
        with open(fn_similarity, 'wb') as f:
            pkl.dump(SimilarityArr, f)
        
    
    # Write to files
    Cleandf.to_csv(fn_cleand, index=False)
    with open(fn_filter, 'wb') as f:
        pkl.dump(FilterDic, f)

