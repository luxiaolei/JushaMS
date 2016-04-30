
import jushacore as mapper 
import numpy as np
import json
import pickle as pkl
import os
import sys

def to_d3js_graph(mapper_output, fn, baseDir, genJson):
    """
    Convert the 1-skeleton of a L{mapper_output} to a dictionary
    designed for exporting to a json package for use with the d3.js
    force-layout graph visualisation system.
    
    *Pram mapper_output: mapper_output object
    *Pram fn: file path to save result
    *Pram genJson: genrates json file with filename=fn if True, else return result
    """
    #construct nodes postition
    S = mapper_output.simplices
    Nodes = mapper_output.nodes
    vertices, vertex_pos = mapper.tools.graphviz_node_pos(S, Nodes)
    pos_dic = {}
    for node_index, pos in zip(vertices, vertex_pos):
        pos_dic[node_index] = pos
    
    G = {}
    G['vertices'] = [{'index': i, 
                      'members': [int(i) for i in list(n.points)], 
                      'attribute': n.attribute, 
                      'pos': pos_dic[i]
                     }
                     for (i,n) in enumerate(Nodes)]
    G['edges'] = [{'source': e[0], 'target': e[1], 'wt':
                   S[1][e]} for e in S[1].keys()]
    
    if genJson:
        baseDir = os.path.join(baseDir,'results')
        fpath = os.path.join(baseDir, fn)
        with open(fpath+'.json', 'w') as f:
            json.dump(G, f)
        print('Progress, {0}.json generation done!'.format(fn))
    else:
        return G

def genIODic(interval, overlap):
    #construct meshgrid for interval and overlap
    xx, yy = np.meshgrid(interval, overlap)
    itNop = zip(xx.ravel(), yy.ravel())
    
    Dic = {}
    for k, tup in enumerate(itNop):
        fn = 'i'+str(tup[0])+'o'+ str(tup[1])
        Dic[fn] = tup
    return Dic


class progressPrinter:
    def __init__(self, previousStep, gap, v=True):
        self.steps = previousStep
        self.gap = gap
        self.v = v
    @property
    def printStep(self):
        if self.v:
            self.steps += self.gap
            if self.steps < 1.:
                assert self.steps < 1
                print('<<<{0:.2f}>>>'.format(self.steps))
                sys.stdout.flush()

def minNodeDiameter():
    return 100
