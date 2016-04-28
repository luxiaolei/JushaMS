# -*- coding: utf-8 -*-
"""
#/filters.py

provides filter functions
"""
import os
import pandas as pd
import numpy as np
import pickle as pkl
from itertools import combinations
from preporcess import CleanerAnny

from tools import progressPrinter

class FilterKnowledge:
    """
    filters which info are extracted from rawdata knowledge
    """
    def __init__(self, confUser):
        self.confUser = confUser
        self.datadir = self.confUser['DATADIR']
        self.filterDicDir = os.path.join(self.datadir, 'filterDic.pkl')

        fn_dfX = os.path.join(self.datadir, 'transformed.csv')
        self.arrX = pd.read_csv(fn_dfX).values

        self.pgPrinter = progressPrinter(0, .1)
        #self.pgPrinter.printStep
        

    def assetFilterDic(self, dim= 2 ,intresedCode= [107, 130, 170], genF= True):
        """
        Returns a dic of filters. 
        keys:f1_107 for dim=1,  f2_107_101 for dim=2.
        values: np.ndarry with shape[1]==dim

        Param dim: dimension of the returned array
        Param intresedCode: elements used to search for asset code in transformed.csv
        Param genF: if True, gen filters and save to filterDic.pkl
        """
        #generates the intresed Dataframe
        cleannerAnny = CleanerAnny(self.confUser, ['金融资产'])
        dfYall = cleannerAnny.startCleaning(intresedCode= intresedCode)
        self.pgPrinter.printStep

        maxdim = dfYall.shape[1] -1
        assert dim <= maxdim

        if genF:
            #save filtDic to pkl
            return self.__genDic(dim, dfYall)
            self.pgPrinter.printStep
        else:
            with open(self.filterDicDir, 'rb') as f:
                filterDic = pkl.load(f)
            return filterDic


    def __genDic(self, dim, dfYall):
            filterDic = {}
            if dim ==1:
                #1-d filter
                for c in dfYall.columns[1:]: #exclude uid column
                    ##construct value
                    arrY = dfYall[c].values
                    arrSVCdis = self.__SVCdis(self.arrX, arrY)
                    
                    ##construct key name
                    #parse exsiting code name, e.g: 金融资产代码_101 ==> f1_101
                    key = c.split('_')[-1]
                    key = 'f1_'+key
                    filterDic[key] = arrSVCdis
            elif dim == 2:
                #2-d filter
                cs = dfYall.columns[1:3].values
                
                ks, vs = [], []
                for c in dfYall.columns[1:]: #exclude uid column
                    ##construct value
                    arrY = dfYall[c].values
                    arrSVCdis = self.__SVCdis(self.arrX, arrY)
                    
                    ##construct key name
                    #parse exsiting code name, e.g: 金融资产代码_101 ==> f1_101
                    key = c.split('_')[-1]
                    key = '_'+key
                    
                    ks.append(key)
                    vs.append(arrSVCdis)
                
                #do combinations
                kComb = list(combinations(ks, 2)) #[(k1, k2), (k1, k3), (k2, k3)]
                vComb = list(combinations(vs, 2))
                for k, v in zip(kComb, vComb):
                    v2d = np.vstack((v[0], v[1])).T
                    assert v2d.shape[1] == 2
                    k2d = 'f2'+ k[0]+ k[1]
                    filterDic[k2d] = v2d
            else:
                ValueError
            with open(self.filterDicDir, 'wb') as f:
                pkl.dump(filterDic, f)
            return filterDic
    
    def __SVCdis(self, arrX, arrY):
        """
        return the distance to hyperplane with optimal parameters
        """
        from sklearn import svm
        #arrY = np.random.random(arrX.shape[0])
        
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        self.pgPrinter.printStep
        
        clf.fit(arrX)
        self.pgPrinter.printStep

        dis2hyperplane = clf.decision_function(arrX).T[0]
        self.pgPrinter.printStep

        #implement cross validates
        return dis2hyperplane

if __name__=="__main__":
    from __init__ import *
    Filter = FilterKnowledge(confUser)
    res = Filter.assetFilterDic()


