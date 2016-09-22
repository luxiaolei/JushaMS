"""
mainTopoGraphs.py
>>>>>>>>>>Run command from terminal:
* python3 mainTopoGraphs.py $DATA_DIR
"""
import sys
import gc
import time
from datetime import datetime
from os import path
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

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

ioPool = ThreadPoolExecutor(max_workers = 2)
computePool = ThreadPoolExecutor(max_workers = 4)

dataDir = sys.argv[1]
resultsDir = path.join(dataDir, 'results')
metaJsonFile = path.join(dataDir, 'metadata.json')

metaJson = json.load(open(metaJsonFile, 'r', encoding = 'utf8'))

params = json.load(open(path.join(dataDir, 'params.json'), 'r', encoding = 'utf8'))
params.sort(key = lambda x: (x['assetCode'], x['interval'], x['overlap']))

filters = pkl.load(open(path.join(dataDir, 'FilterDic.pkl'), 'rb'))

dist_matrix = np.load(open(path.join(dataDir, 'dist_matrix.npy'), 'rb'))

gc.collect()

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
    idx = [i for i, x in enumerate(metaJson['results']) if x.keys().__contains__(name)][0]
    metaJson['results'][idx][name] = status
    with open(metaJsonFile, 'w', encoding = 'utf8') as f:
        json.dump(metaJson, f, ensure_ascii = False, sort_keys = True, indent = 4)

def update_file_progress(file_name, p):
    print_msg('!!!!!{0}: {1:.2f}!!!!!'.format(file_name, p))
    update_metadata(file_name, p)

def save_topo_graph(mapper_output, filter, cover, file_name):
    global resultsDir
    mapper.scale_graph(mapper_output, filter, cover = cover, weighting = 'inverse',
                       exponent = 1, verbose = True)
    update_file_progress(file_name, 0.9)
    status = 0
    if mapper_output.stopFlag:
        print_msg(file_name + ' Stopped! Too many nodes or too long time')
        status = -1
    else:
        to_d3js_graph(mapper_output, file_name, resultsDir, True)
        status = 1
    update_file_progress(file_name, status)
    return status

def core_wrapper(interval, overlap, assetCode, file_name):
    global resultsDir
    global dist_matrix
    global filters
    filter = filters[assetCode]
    print_msg('Data shape: ' + str(dist_matrix.shape) + ', Filter shape: ' + str(filter.shape))
    update_file_progress(file_name, 0)
    cover = mapper.cover.cube_cover_primitive(interval, overlap)
    mapper_output = mapper.jushacore(dist_matrix, filter, cover = cover, cutoff = None,
                                     cluster = mapper.single_linkage(),
                                     metricpar = { 'metric': 'euclidean' },
                                     verbose = True)
    update_file_progress(file_name, 0.5)
    gc.collect()
    return ioPool.submit(save_topo_graph, mapper_output, filter, cover, file_name)

#####################################
##          RESULTS                ##
#####################################

p = 0.3
print_msg('<<<<<Progress[results]: {0:.2f}>>>>>'.format(p))

step = (0.8 - p) / len(params)
future_to_file = {}
for param in params:
    (i, o, a) = (param['interval'], param['overlap'], param['assetCode'])
    file_name = 'i' + str(i) + 'o' + str(o) + a
    f = computePool.submit(core_wrapper, i, o, a, file_name)
    future_to_file[f] = file_name
    p += step
    print_msg('<<<<<Progress[results]: {0:.2f}>>>>>'.format(p))
    time.sleep(int(len(list(filters.items())[0][1]) / 100000.0 * 100))
step = (0.98 - p) / len(params)
future_to_file_status = {}
for f in futures.as_completed(future_to_file):
    file_name = future_to_file[f]
    status = 0
    try:
        future = f.result()
    except Exception as ex:
        status = -1
        print_msg('Result %r generated an exception: %s' % (file_name, ex))
        update_file_progress(file_name, status)
    else:
        future_to_file[future] = (file_name, status)
    p += step
    print_msg('<<<<<Progress[results]: {0:.2f}>>>>>'.format(p))
for f in futures.as_completed(future_to_file_status):
    (file_name, status) = future_to_file_status[f]
    if status != -1:
        try:
            status = f.result()
        except Exception as ex:
            status = -1
            print_msg('Result %r generated an exception: %s' % (file_name, ex))

print_msg('<<<<<Progress[results]: 1>>>>>')
