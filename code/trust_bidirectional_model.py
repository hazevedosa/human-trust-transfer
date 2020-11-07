# our imports
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import Parameter

import numpy as np
from numpy.linalg import norm

import scipy.io as sio

dtype = torch.FloatTensor

class BidirectionalTrustModel(torch.nn.Module):

    # Init Method (define parameters)
    def __init__(
                self, 
                modelname, 
                inpsize, 
                obsseqlen,
                taskrepsize,
                capabilityRepSize
                ):
        super(BidirectionalTrustModel, self).__init__()

        self.capabilityMean = Variable(dtype(np.zeros((capabilityRepSize,1))), requires_grad=False)
        self.capabilityCovariance = Variable(dtype(1e5 * np.eye((capabilityRepSize))), requires_grad=False)

        self.trustPropensity = Parameter(dtype(np.eye(1)))

    

    # Forward Method (model process)
    def forward(self, inptasksobs, inptasksperf, inptaskspred, num_obs_tasks):

        observationsNumber  = inptasksobs.shape[0]
        predictionsNumber   = inptasksobs.shape[1]

        predictedTrust      = Variable(dtype(np.zeros((predictionsNumber, 1))), requires_grad=False)


        # Capabilities estimation loop
        for i in range(observationsNumber):
            self.capabilityUpdate()

        # Trust computation loop
        for i in range(predictionsNumber):
            predictedTrust[i] = self.computeTrust(i)

        obsTrust = predictedTrust
        return obsTrust


    # Auxiliary Methods
    def capabilityUpdate(self):



        self.capabilityMean = 
        self.capabilityCovariance = 

        return

    def requirementTransform(self):
        return
    
    def computeTrust(self, i):
        return i




# Non-class methods
def createDataset_fromMatFile(mat_file_name):

    mat_contents = sio.loadmat(mat_file_name)

    tasksobsfeats   = mat_contents["tasksobsfeats"]
    tasksobsperf    = mat_contents["tasksobsperf"]
    taskspredfeats  = mat_contents["taskspredfeats"]
    trustpred       = mat_contents["trustpred"]
    tasksobsids     = mat_contents["tasksobsids"]
    taskpredids     = mat_contents["taskpredids"]
    taskpredtrust   = mat_contents["taskpredtrust"]
    matTasks        = mat_contents["matTasks"]
    matTaskPredIDs  = mat_contents["matTaskPredIDs"]
    data_labels     = ['0-0', 'H-1', 'H-2', 'H-3', 'H-4', 'H-5']

    trustpred = np.squeeze(trustpred)
    tasksobsids = np.expand_dims(tasksobsids, axis=2)

    dataset = (
                tasksobsfeats,   # (51, 255, 50) [numpy.ndarray]
                tasksobsperf,    # (51, 255, 2)  [numpy.ndarray]
                taskspredfeats,  # (255, 50)     [numpy.ndarray]
                trustpred,       # (255,)        [numpy.ndarray]
                tasksobsids,     # (51, 255, 1)  [numpy.ndarray]
                taskpredids,     # (255, 1)      [numpy.ndarray]
                taskpredtrust,   # (255, 1)      [numpy.ndarray]
                matTasks,        # (51, 51)      [numpy.ndarray]
                matTaskPredIDs,  # (55,  5)      [numpy.ndarray]
                data_labels      # ????????????  [list]
    )

    return dataset



# Just for testing...
if __name__ == "__main__":

    mat_file_name = 'RawDataset.mat'
    dataset = createDataset_fromMatFile(mat_file_name)

    TestModel = BidirectionalTrustModel("testName",
                                        dataset[0].shape[2],
                                        dataset[0].shape[0],
                                        dataset[0].shape[2],
                                        3)


    result = TestModel(dataset[0], dataset[1], dataset[2], dataset[0].shape[0])