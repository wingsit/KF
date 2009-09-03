from formatConverter import *
import timeSeriesFrame, mdp
from regression import Regression
from numpy import array
from scipy import matrix, shape

full = FormatConverter("Book1.csv").readCSV().toTSF()
fi = FormatConverter("FI.csv").readCSV().toTSF()
fi = fi[:,0]
#f = full[:,0:6]
f = full
data = f.data
#data = (f.data - mdp.numx.mean(f.data, 0))/mdp.numx.std(f.data, 0)
#data = scipy.transpose(data)
#flow = mdp.Flow([mdp.nodes.NIPALSNode(output_dim=4),mdp.nodes.CuBICANode()], verbose = 1)
flow = mdp.Flow([mdp.nodes.PCANode(output_dim=4)], verbose = 1)

#flow = mdp.nodes.PCANode(output_dim = 7)
#flow.set_crash_recovery(1)
print "shape(data): ", shape(data)
flow.train(array(data))
#flow.stop_training(debug = True)
output = flow(data)
import code; code.interact(local=locals())

g = timeSeriesFrame.TimeSeriesFrame(output, f.rheader, ["PCA1", "PCA2", "PCA3"])
import code; code.interact(local=locals())

print fi.size()
print g.size()
t,n = g.size()
weight = scipy.identity(t)
obj = Regression(fi, g, intercept = False, weight = weight)
obj.train()
print obj.R2()

g = full[:, 0]
print fi.size()
print g.size()
t,n = g.size()
weight = scipy.identity(t)
obj = Regression(fi, g, intercept = False, weight = weight)
obj.train()
print obj.R2()
print g
print fi
