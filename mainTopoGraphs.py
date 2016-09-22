"""
mainTopoGraphs.py
>>>>>>>>>>Run command from terminal:
* python3 mainTopoGraphs.py $DATA_DIR
"""
import sys
import gc
from datetime import datetime
from os import path

import json
import pickle as pkl
import numpy as np

import jushacore as mapper
from tools import to_d3js_graph

#####################################
##          GLOBAL VAR             ##
#####################################

start_at = datetime.now()
print('================ Start at ' + str(start_at) + ' ================', flush = True)

dataDir = sys.argv[1]
resultsDir = path.join(dataDir, 'results')
metaJsonFile = path.join(dataDir, 'metadata.json')

metaJson = json.load(open(metaJsonFile, 'r', encoding = 'utf8'))

params = json.load(open(path.join(dataDir, 'params.json'), 'r', encoding = 'utf8'))
params.sort(key = lambda x: (x['assetCode'], x['interval'], x['overlap']))

filters = pkl.load(open(path.join(dataDir, 'FilterDic.pkl'), 'rb'))

dist_matrix = np.load(open(path.join(dataDir, 'dist_matrix.npy'), 'rb'))

#####################################
##          FUNCTIONS              ##
#####################################

def print_msg(msg):
    elapsed_time_msg = '========= Elapsed time: {0:.2f} sec =========\n'.format((datetime.now() - start_at).total_seconds())
    print(elapsed_time_msg + msg + '\n', end = '', flush = True)

def update_metadata(file_name, status):
    global metaJsonFile
    global metaJson
    name = file_name + '.json'
    idx = [i for i, x in enumerate(metaJson['results']) if x[name] <= 0][0]
    metaJson['results'][idx][name] = status
    with open(metaJsonFile, 'w', encoding = 'utf8') as f:
        json.dump(metaJson, f, ensure_ascii = False, sort_keys = True, indent = 4)

def core_wrapper(interval, overlap, assetCode, file_name):
    global resultsDir
    global dist_matrix
    global filters
    filter = filters[assetCode]
    print_msg('Data shape: ' + str(dist_matrix.shape) + ', Filter shape: ' + str(filter.shape))
    print_msg('!!!!!' + file_name + ': 0!!!!!')
    try:
        cover = mapper.cover.cube_cover_primitive(interval, overlap)
        mapper_output = mapper.jushacore(data, filter, cover = cover, cutoff = None,
                                         cluster = mapper.single_linkage(),
                                         metricpar = { 'metric': 'euclidean' },
                                         verbose = False)
        print_msg('!!!!!' + file_name + ': 0.3!!!!!')
        gc.collect()
        mapper.scale_graph(mapper_output, filter, cover = cover, weighting = 'inverse',
                           exponent = 1, verbose = False)
        print_msg('!!!!!' + file_name + ': 0.6!!!!!')
        gc.collect()
    except Exception:
        return -1
    else:
        if mapper_output.stopFlag:
            print_msg(file_name + ' Stopped! Too many nodes or too long time')
            return -1
        else:
            to_d3js_graph(mapper_output, file_name, resultsDir, True)
            print_msg('!!!!!' + file_name + ': 0.9!!!!!')
            gc.collect()
            return 1

#####################################
##          RESULTS                ##
#####################################

p = 0.3
print_msg('<<<<<Progress[results]: {0:.2f}>>>>>'.format(p))

step = (1 - p) / len(params)
for param in params:
    (i, o, a) = (param['interval'], param['overlap'], param['assetCode'])
    file_name = 'i' + str(i) + 'o' + str(o) + a
    try:
        status = core_wrapper(i, o, a, file_name)
    except Exception as ex:
        status = -1
        print('Result %r generated an exception: %s' % (file_name, ex))
    print_msg('!!!!!' + file_name + ': ' + str(status) + '!!!!!')
    update_metadata(file_name, status)
    p += step
    print_msg('<<<<<Progress[results]: {0:.2f}>>>>>'.format(p))

print_msg('<<<<<Progress[results]: 1>>>>>')
