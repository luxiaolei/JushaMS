"""
mainResults.py

>>>>>>Inputs:

* data/session/FilterDic.pkl 
    dict type with key '_107', '_170', '_130'. values are 1-D array

* data/session/_107_data.pkl
* data/session/_170_data.pkl
* data/session/_130_data.pkl
    values are 1-D array with length (n_samplesize * n_sample)/2 - n_samplesize


>>>>>>Functionalities:
* Run TDA with inputs params

>>>>>>Produce:
* data/session/results/*.json

>>>>>>Run command from terminal:
* python3 mainResults.py data/session interval overlap assetKey
    
"""


from __init__ import confUser
import pandas as pd
import pickle as pkl
import jushacore as mapper 
from tools import to_d3js_graph, progressPrinter
import sys
import os


def core_wrapper(baseDir, data, filt, interval, overlap, fn, genJson= True):
    """
    class method of class coreWrapper. it recieves inputs required by jushacore as parameters
    and return the result if genJson==False, else write result to fn.
    """
    #assert data.shape[0] == filt.shape[0]
    print(data.shape, filt.shape, interval, overlap, fn)
    
    cluster = mapper.single_linkage()
    cover = mapper.cover.cube_cover_primitive(interval, overlap)
    metricpar = {'metric': 'euclidean'}

    mapper_output = mapper.jushacore(data, filt, cover=cover,\
                             cutoff=None, cluster=cluster,\
                             metricpar=metricpar, verbose=True)
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
        to_d3js_graph(mapper_output, fn, baseDir, genJson)
        print('Core ran finished! with: {0}'.format(fn))


if __name__=='__main__':

    #dataDir = 'data/tempdir'# sys.argv[1]
    dataDir = sys.argv[1]
    interval = sys.argv[2]
    overlap = sys.argv[3]
    filtKey = sys.argv[4]  #inputs: _107



    #dataDir = confUser['DATADIR']
    fn_data = os.path.join(dataDir, filtKey+'_data.pkl')
    fn_filter = os.path.join(dataDir, 'FilterDic.pkl')
    fn_results = os.path.join(dataDir, 'results')
    
    n = .33
    print('<<<{0:.2f}>>>'.format(1.*n))
    sys.stdout.flush()

    with open(fn_filter, 'rb') as f:
        FilterDic = pkl.load(f)
    with open(fn_data, 'rb') as f:
        data = pkl.load(f)

    print('<<<{0:.2f}>>>'.format(2.*n))
    sys.stdout.flush()
    filt = FilterDic[filtKey]
    fn = 'i'+interval+'o'+overlap+filtKey
    
    core_wrapper(fn_results, data, filt, int(interval), int(overlap), fn)
    pgPrinter.printStep
    sys.stdout.flush()
    print('<<<1>>>')

