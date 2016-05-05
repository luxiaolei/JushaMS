# -*- coding: utf-8 -*-

"""

#/wrapper.py


"""

import pandas as pd
import numpy as np
import scipy
import json
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
import os
import time

from tools import genIODic

t0 = time.time()


class coreWrapper:
    """
    
    """
    def __init__(self, confUser):
        """
        *Param confUser: Dic stores user-level params
        """
        self.interval = confUser['interval']
        self.overlap = confUser['overlap']
        self.datadir = os.path.join(confUser['DATADIR'], 'results')
        pass
    
  
    def assetStartWrapping(self, FilterKnowledge, dim=2, intresedCode= [107, 130, 170]):
        """
        *Param FilterKnowledge: FilterKnowledge instance
        """
        
        #constrcut parameters generator for multithreading
        # filterDic.pkl mush already exsit!
        filterDic = FilterKnowledge.assetFilterDic(dim, intresedCode= intresedCode, genF= False)
        paramDic = genIODic(self.interval, self.overlap)
        
        data = FilterKnowledge.arrX
        parmsGenerator = self.__inputsGenerator(data, filterDic, paramDic)
        
        pool = ThreadPool()
        results = pool.map(self.__multi_run_wrapper, parmsGenerator)
        pool.close()
        pool.join()
        
        
    def __multi_run_wrapper(self, args):
        print('Unpacking params, for epoch:',args[-1])
        
        return self.core_wrapper(*args)
    
    #@classmethod
    def core_wrapper(self, data, filt, interval, overlap, fn, genJson= True):
        """
        class method of class coreWrapper. it recieves inputs required by jushacore as parameters
        and return the result if genJson==False, else write result to fn.
        """
        assert data.shape[0] == filt.shape[0]
        print(data.shape, filt.shape, interval, overlap, fn)
        
        cluster = mapper.single_linkage()
        cover = mapper.cover.cube_cover_primitive(interval, overlap)
        metricpar = {'metric': 'euclidean'}

        mapper_output = mapper.jushacore(data, filt, cover=cover,\
                                 cutoff=None, cluster=cluster,\
                                 metricpar=metricpar, verbose=False)
        mapper.scale_graph(mapper_output, filt, cover=cover, weighting='inverse',\
                      exponent=1, verbose=False)
        if mapper_output.stopFlag:
            print('{0} Stopped! Too many nodes or too long time'.format(fn))
        else:
            print('{0} Successed! '.format(fn))
            print('type check! ',type(mapper_output))
            import pickle as pkl
            with open('G.pkl', 'wb') as f:
                pkl.dump(mapper_output, f)

            to_d3js_graph(mapper_output, fn, self.datadir, genJson)
            print('Core ran finished! with: {0}'.format(fn))
            #return mapper_output
    
    def __inputsGenerator(self, data, filterDic, paramDic):
        """
        It yields a tuple of inputs used for multiThreading 
        
        *Param data: data array for core
        *Type: np.nDarry, with no missing value, and np.float or np.int types for each data pt
        
        *Return generator: tuple of all the inputs.
        """
        assert isinstance(data, np.ndarray) 
        assert not np.isnan(data).sum() #also make sure no none-number in data
        
        ans = []
        for kp, vp in paramDic.items():
            for kf, vf in filterDic.items():
                #construct json name with structure. kp_kf e.g., i10o75_f2_107_108 or i10o75_f1_107
                 yield (data, vf, vp[0], vp[1], kp+ '_'+ kf)
                
                #? ans.append((data, vf, vp[0], vp[1], kp+ '_'+ kf))
        #return ans

    def __coreInputsChecker(*args):
        """
        check inputs for jushacore, print error if appears.
        """
        pass



    

if __name__=='__main__':
    #@debugging record
    #single run mapper no problem
    
    from __init__ import *
    import jushacore as mapper 
    from filters import FilterKnowledge
    from tools import to_d3js_graph, minNodeDiameter
    #from preporcess import CleanerAnny
    
    t0 = time.time()
    KnowlgeF = FilterKnowledge(confUser)
    
    coreWrapper = coreWrapper(confUser)
    coreWrapper.assetStartWrapping(KnowlgeF, dim=2, intresedCode= [107, 130, 107]) #107，130，170

    t1 = time.time()
    print('for 11596,26 data, Dimension {0}, total time cost {1} s'.format(2, t1-t0))
    


