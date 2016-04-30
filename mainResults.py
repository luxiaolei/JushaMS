


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
    assert data.shape[0] == filt.shape[0]
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
	filtKey = sys.argv[4]
	print(type(interval))
	pgPrinter = progressPrinter(-.2, .2)
	pgPrinter.printStep
	sys.stdout.flush()


	#dataDir = confUser['DATADIR']
	dataTransDir = os.path.join(dataDir, 'transformed.csv')
	data = pd.read_csv(dataTransDir).values
	pgPrinter.printStep
	sys.stdout.flush()

	filtDir = os.path.join(dataDir, 'filterDic.pkl') 
	pgPrinter.printStep
	sys.stdout.flush()

	print(filtDir)

	with open(filtDir, 'rb') as f:
		filtDic = pkl.load(f)
	print(filtDic.keys())
	filt = filtDic[filtKey]
	pgPrinter.printStep
	sys.stdout.flush()

	fn = 'i'+interval+'o'+overlap+filtKey

	core_wrapper(dataDir, data, filt, int(interval), int(overlap), fn)
	pgPrinter.printStep
	sys.stdout.flush()
	print('<<<1>>>')




	



