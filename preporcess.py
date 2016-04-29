# -*- coding: utf-8 -*-
"""
/preporcess.py

* clean raw data. Do the following, encode, concate, replace null.
* data checker. makesure the data, contains no null value, type is correct. 
"""
import sys
import os
import pandas as pd
import numpy as np
import json

from tools import progressPrinter


__all__ = ['Cleaner', 'CleanerAnny']



class Cleaner:
    """
    This class prepare the data for transfomation.
    """
    def __init__(self, confUser):
        """
        *param confUser, stores config values for USER sector
        *type confUser, configparser.SectionProxy
        """
        self.datadir = confUser['DATADIR']
        self.datain = os.path.join(self.datadir, 'rawdata') #str
        self.dataout = os.path.join(self.datadir, 'result') #str
        self.keyword = confUser['KEYWORD'] #list of keywords
        self.deli = confUser['DELI'] #str
        self.encoding = confUser['ENCODING']
        self.nanvalue = confUser['NANVALUE']
        self.ext = confUser['EXTENSION']
        
        self.keyword = self.keyword.split(',') #the first keyword is primary
        self.nanvalue = self.nanvalue.split(',')
        
        self.fnsPath = []
        self.cleanedDf = []
        self.error = []
        self.errorTrace = []
        print(self.datain)

        
        

        
    
    def startCleaning(self):
        """
        do the cleannig processes consecutively
        """
        self.pgPrinter = progressPrinter(-.1, .1)
        self.pgPrinter.printStep

        self._kw2Dfs = self._getDfs(1)
        self.cleanedDf = self._getDfs(0)
        print('Dataframes are ready!')
        self.pgPrinter.printStep
        
        self.pgPrinter.printStep
        self.pgPrinter.printStep
        self.__concateNjoin()
        print('Concate done!')
        self.pgPrinter.printStep
  
        assert self.cleanedDf.shape[0] > 100  
        #assert not self.error, print(self.error)
        self.pgPrinter.printStep
        #debug
        if not self.error: print(self.error)
        
        fnc = os.path.join(self.datadir, 'cleaned.csv')
        self.cleanedDf.to_csv(fnc, index=True, index_label='uid')
        self.pgPrinter.printStep
        self.pgPrinter.printStep
              


    def startTransform(self, mpDir, ucols):
        #start do transfomations, for jushacore
        #fillna with 0, and select useful features
        self.pgPrinter = progressPrinter(-.1, .1)
        self.pgPrinter.printStep


        fnc = os.path.join(self.datadir, 'cleaned.csv')
        cleanedDf = pd.read_csv(fnc)
        self.tranDf = cleanedDf.fillna(0)
        self.tranDf = self.tranDf.ix[:, ucols]
        print('nan filled and df sub selected!')
        self.pgPrinter.printStep
        
        self.__mapping(mpDir)
        print('data mapping done!')
        self.pgPrinter.printStep
        
        self.__outlierTreat()
        print('data outlier treated!')
        self.pgPrinter.printStep
        
        self.__standardize()
        print('data standardization done!')
        self.pgPrinter.printStep

        #save to transformed.csv adn uid to uid.csv
        assert self.tranDf.shape[0] > 100
        fnt = os.path.join(self.datadir, 'transformed.csv')
        self.tranDf.to_csv(fnt, index=False) #set index=False lose uid info
        self.pgPrinter.printStep
        
        fnid = os.path.join(self.datadir, 'uid.csv')
        uidf = pd.DataFrame(self.tranDf.index, columns=['uid'])
        uidf.to_csv(fnid, index=False)
        print('Files are generated!')
        self.pgPrinter.printStep

    
         
    def _getDfs(self, p=0):
        """
        extract filepath which statisfy keyword&ext conditions
        return list of df
        *args 
        """
        paths = []
        for dirpath, dirnames, filenames in os.walk(self.datain):
            for fn in filenames:
                if self.keyword[p] in fn and fn.endswith(self.ext):
                    #fn has pattern: *keword*.txt*
                    fnPath = os.path.join(dirpath, fn)
                    paths.append(fnPath)
        
        #read_csv and catch errors
        assert len(paths)> 1
        dfs = []
        for k, fn in enumerate(paths):
            try:
                df = pd.read_csv(fn, delimiter= self.deli, encoding='gb18030', 
                                 na_values= self.nanvalue, error_bad_lines= False)
                #未知 can not be accepted by pandas
                df.replace(['未知', '其他', '?', ' '], np.nan, inplace=True)

                #for p=1, '\t' is also a delimiter, special case
                if df.shape[1] < 2:
                    df = pd.read_csv(fn, delimiter='\t', encoding='gb18030', 
                                     na_values= self.nanvalue, error_bad_lines= False)

                if k> 0: assert df.shape[1]== dfs[k-1].shape[1]
                dfs.append(df)
            
            except Exception as e:
                msg = 'file: '+ fn+ ', error in getting file porcess!'
                self.error.append(msg)
                self.errorTrace.append(e)
            
        return dfs 

    def __concateNjoin(self):
        """
        concates files based on KEYWORD, recursely in DATAIN dir.
        drop duplicated rows based on uid, and set uid as index.
        inner join userimage df and correspondingInstitude df.
        """
        #concate dfs
        try:
            self._kw2Dfs = pd.concat(self._kw2Dfs)
            self.cleanedDf = pd.concat(self.cleanedDf)
            self.pgPrinter.printStep
        except Exception as e:
            msg = 'error in concating Dataframes process!'
            self.error.append(msg)
            self.errorTrace.append(e)

        #drop duplicated user id, #and set uid to be index
        try:
            self.cleanedDf.drop_duplicates(subset='客户代码', inplace= True)
            self._kw2Dfs.drop_duplicates(subset='khdm', inplace= True)
            self.pgPrinter.printStep
            #self.cleanedDf.set_index('客户代码', drop=True, inplace=True)
            #self._kw2Dfs.set_index('khdm', drop=True, inplace=True)
        except Exception as e:
            msg = 'error in drop_duplicate uid and set_index processes!'
            self.error.append(msg)
            self.errorTrace.append(e)
            
        #inner join on uid
        try:
            self.cleanedDf = pd.concat([self._kw2Dfs, self.cleanedDf], axis=1, join='inner')
            self.pgPrinter.printStep
        except Exception as e:
            msg = '无法合并用户画像表以及用户对应机构表'
            self.error.append(msg)
            self.errorTrace.append(e)
            self.pgPrinter.printStep
    
    def __mapping(self, mpDir):
        """
        maps data according to maptable
        """
        maptable = pd.read_csv(mpDir, header=None)#, encoding='gbk')
        maptable = maptable.ix[:, :1].values
        for k,v in maptable:
            try:
                self.tranDf.replace(k,v, inplace=True)
            except Exception as e:
                msg = '数值映射时发生错误!'
                self.error.append(msg)
                self.errorTrace.append(e)
        self.pgPrinter.printStep
                
    def __outlierTreat(self):
        self.tranDf['年龄'] = self.tranDf['年龄'].apply(lambda x: x if x < 120 else 120 )
        self.pgPrinter.printStep
    
    def __standardize(self):
        for k, dtp in enumerate(self.tranDf.dtypes):
            try:
                assert dtp == np.int or dtp == np.float
            except Exception as e:
                msg = '尝试标准化数据时，发现有非数字的列: '+ self.tranDf.columns[k]
                self.error.append(msg)
                self.errorTrace.append(e)
        self.pgPrinter.printStep
                
        from sklearn import preprocessing
        v = preprocessing.StandardScaler().fit_transform(self.tranDf)
        self.tranDf = pd.DataFrame(v, columns=self.tranDf.columns, 
                               index= self.tranDf.index)
        self.pgPrinter.printStep
    
#This class respondes for cleanning data used for generating filters 
class CleanerAnny(Cleaner):
    """
    Cleaner for cleanning asset_table
    """
    def __init__(self, confUser, confKw):
        Cleaner.__init__(self, confUser)
        self.keyword = confKw

        #self.pgPrinter = progressPrinter(0, .25)
        #self.pgPrinter.printStep
        
    def startCleaning(self, intresedCode= [107, 101]):
        #read uid.csv into an array
        fnuid = os.path.join(self.datadir, 'uid.csv')
        self.uidArr = pd.read_csv(fnuid).values
        print(self.uidArr.shape[0])
        #self.pgPrinter.printStep
        
        #read raw files and concate them
        dfList = self._getDfs(p=0)
        assetDf = pd.concat(dfList, axis=0)
        print(assetDf.shape)
        #self.pgPrinter.printStep
        
        
        assetDf = self.__assetCode(assetDf, intresedCode)
        print(assetDf.shape)
        #self.pgPrinter.printStep
        return assetDf
    
    def __assetCode(self, assetDf, intresedCode):
        #select 金融资产代码 and 核心客户号 columns
        #select rows based on intresed 金融资产代码
        assetDf = assetDf.ix[assetDf['金融资产代码'].isin(intresedCode),
                             ['核心客户号', '金融资产代码']]

        #drop duplicates on subset=['核心客户号','金融资产代码']
        assetDf.drop_duplicates(subset=['核心客户号','金融资产代码'], inplace=True)

        #make sure assetDf's uid is the subset of userimage's uid.
        assetDf_trimed = assetDf.ix[assetDf['核心客户号'].isin(self.uidArr.ravel()), :]

        #dummy coded and compress alone uid column(so that uids are distict)
        assetDf_trimed_dummy = pd.get_dummies(assetDf_trimed, columns=['金融资产代码'])
        dumyFixed = []
        col = assetDf_trimed_dummy.columns.values
        col = col[col != '核心客户号']
        for uid, df in assetDf_trimed_dummy.groupby('核心客户号'):
            row = df.ix[:, col].values
            if df.shape[0] > 1:
                row = row.sum(axis= 0)
            row = np.insert(row, 0, uid)
            dumyFixed.append(row)  
        assetDf_trimed_dummy = pd.DataFrame(dumyFixed, columns=np.insert(col, 0, '核心客户号'))

        #construct target dataframe by left join uid and assetDf.
        targetDf = pd.DataFrame(self.uidArr, columns=['核心客户号'])
        targetDfnew = pd.merge(targetDf, assetDf_trimed_dummy, on='核心客户号', how='left')
        
        #fill those users who didnt buy anything intresedCode product, with 0
        targetDfnew.fillna(0, inplace=True)
        return targetDfnew
        

        
        
if __name__=="__main__":
    """
    #print(sys.argv[1])

    #recieve sys.arg[1] as basedir in data/
    #set config.ini DATAIN & DATAOUT value
    config = configparser.ConfigParser()
    config.read('../config/config.ini')
    confUser = config['USER']
    confUser['DATADIR'] = '../data/tempdir'# 'data/'+ sys.arg[1]
    maptableDir = '../config/mappingTable.csv'
    usefulCols = ['性别', '年龄', '婚姻', '学历', '从属行业',
    '理财抗风险等级', '客户层级', '新老客户标记', '五级分类', '小微客户类型', '消费类资产产品', '纯消费性微贷标记',
    '纯质押贷款标记', '非储投资偏好', '高金融资产标记', '购买大额他行理财标记', '大额消费标记', '信用卡高还款标记',
    '信用卡高端消费标记', '优质行业标记', '高代发额客户标记','潜在高端客户标记', '客户贡献度',
    '客户活跃度', '客户渠道偏好', '客户金融资产偏好']
    """
    from __init__ import *
    cleaner = Cleaner(confUser)
    cleaner.startCleaning(maptableDir, usefulCols)
    cleaner.saveNlog()
    print('error msg: ',cleaner.error, cleaner.errorTrace)
    

    cleannerAnny = CleanerAnny(confUser, ['金融资产'])
    print(cleannerAnny.startCleaning().shape)


