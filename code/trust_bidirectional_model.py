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
                capabilityRepresentationSize
                ):
        super(BidirectionalTrustModel, self).__init__()


        self.capabilityRepresentationSize = capabilityRepresentationSize # how many capabilities are represented
        self.capabilityMean = Variable(dtype(np.zeros((self.capabilityRepresentationSize,1))), requires_grad=False) # initialized as zeros

        self.trustPropensity = Parameter(dtype(np.eye(1))) # parameter to be optimized


    # Forward Method (model process)
    def forward(self, inptasksobs, inptasksperf, inptaskspred, num_obs_tasks):

        tasksPerObservationSequence = inptasksobs.shape[0]  # 51 for our dataset
        observationSequencesNumber  = inptasksobs.shape[1]  # 255 for our dataset
        trustPredictionsNumber      = 1                     # adequate to the dataset format...

        predictedTrust              = Variable(dtype(np.zeros((observationSequencesNumber, trustPredictionsNumber))), requires_grad=False) # (255, 1) for our dataset


        # for each (of the 255) observations sequence prior to trust predictions
        for i in range(observationSequencesNumber):
            
            # re-initialize the capability
            self.capabilityMean = Variable(dtype(np.zeros((self.capabilityRepresentationSize,1))), requires_grad=False)


            ## Capabilities estimation loop
            # checks each task on the observation sequence
            for j in range(tasksPerObservationSequence):
                self.capabilityUpdate(inptasksobs[j,i,:], inptasksperf[j,i,:])


            ## Trust computation loop
            # computes trust for each input task... But in our dataset we consider only 1
            for j in range(trustPredictionsNumber):
                # predictedTrust[i, j] = self.computeTrust(inptaskspred[i, j])
                predictedTrust[i, j] = self.computeTrust(inptaskspred[i])

            obsTrust = predictedTrust

        return obsTrust


    # Auxiliary Methods
    def capabilityUpdate(self, observedTask, observedTaskPerformance):

        observedCapability = self.requirementTransform(observedTask)
        taskIsNonZero, taskSuccess = self.getSuccessOrFailBools(observedTaskPerformance)

        if taskIsNonZero:
            if taskSuccess:
                for i in range(self.capabilityRepresentationSize):
                    if observedCapability[i] > self.capabilityMean[i]:
                        self.capabilityMean[i] = observedCapability[i]
            else:
                for i in range(self.capabilityRepresentationSize):
                    if self.capabilityMean[i] > observedCapability[i]:
                        self.capabilityMean[i] = observedCapability[i]
        return

    def requirementTransform(self, observedTask):

        observedCapability = dtype(np.random.rand( self.capabilityRepresentationSize ))

        return observedCapability
    


    def getSuccessOrFailBools(self, observedTaskPerformance):
        
        if not(observedTaskPerformance[0]) and not(observedTaskPerformance[1]):
            taskIsNonZero = False
            taskSuccess = False
        elif not(observedTaskPerformance[0]) and observedTaskPerformance[1]:
            taskIsNonZero = True
            taskSuccess = True
        elif observedTaskPerformance[0] and not(observedTaskPerformance[1]):
            taskIsNonZero = True
            taskSuccess = False
        else:
            print("Error: performance indicators = [1, 1]")
            raise SystemExit(0)

        return taskIsNonZero, taskSuccess
    
    def computeTrust(self, inptaskspred):

        requiredCapability = self.requirementTransform(inptaskspred)

        stepIndicator = 1

        for i in range(self.capabilityRepresentationSize):
            if requiredCapability[i] > self.capabilityMean[i]:
                stepIndicator = stepIndicator * 0

        return stepIndicator * dtype(np.eye(1))




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
                                        2)


    result = TestModel(dataset[0], dataset[1], dataset[2], dataset[0].shape[0])