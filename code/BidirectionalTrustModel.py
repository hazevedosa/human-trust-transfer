# imports
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import Parameter

import numpy as np
from numpy.linalg import norm

import scipy.io as sio

import pickle

usecuda = True
usecuda = usecuda and torch.cuda.is_available()

dtype = torch.FloatTensor

if usecuda:
    dtype = torch.cuda.FloatTensor

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


        self.modelname = modelname

        self.capabilityRepresentationSize = capabilityRepresentationSize # how many capabilities are represented
        self.capabilityMean = Variable(dtype(np.zeros((self.capabilityRepresentationSize,1))), requires_grad=False) # initialized as zeros

        self.betas = Parameter(dtype(20.0 * np.random.rand( capabilityRepresentationSize ))) # parameters to be optimized
        self.zetas = Parameter(dtype(np.random.rand( capabilityRepresentationSize ))) # parameters to be optimized

    # Forward Method (model process)
    def forward(self, inptasksobs, inptasksperf, inptaskspred, num_obs_tasks, tasksobsids, taskspredids):

        

        tasksPerObservationSequence = inptasksobs.shape[0]  # 51 for our dataset // 2 for Soh's
        observationSequencesNumber  = inptasksobs.shape[1]  # 255 for our dataset // 192 or 186 for Soh's
        trustPredictionsNumber      = 1                     # adequate to the dataset format... // (both)
        predictedTrust              = Variable(dtype(np.zeros((observationSequencesNumber, trustPredictionsNumber))), requires_grad=False) 
                                                                                                      # (255, 1) for our dataset // (both)



        # for each (of the 255) observations sequence prior to trust predictions
        for i in range(observationSequencesNumber):
            
            # re-initialize the capability
            self.capabilityMean = Variable(dtype(np.zeros((self.capabilityRepresentationSize,1))), requires_grad=False)


            ## Capabilities estimation loop
            # checks each task on the observation sequence
            for j in range(tasksPerObservationSequence):
                self.capabilityUpdate(inptasksobs[j,i,:], inptasksperf[j,i,:], tasksobsids[j,i,0])


            ## Trust computation loop
            # computes trust for each input task... But in our dataset we consider only 1
            for j in range(trustPredictionsNumber):
                # predictedTrust[i, j] = self.computeTrust(inptaskspred[i, j])
                predictedTrust[i, j] = self.computeTrust(taskspredids[i, 0])

        trust = predictedTrust

        return trust

    # Auxiliary Methods
    def capabilityUpdate(self, observedTask, observedTaskPerformance, observedTaskID):

        observedCapability = self.requirementTransform(observedTaskID)
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

    def requirementTransform(self, observedTaskID):

        capabilitiesMatrix = 0.01 * np.array(   [[0.0, 33.0, 50.0, 43.0, 56.0, 67.0, 62.0, 47.0, 50.0, 51.0, 64.0, 64.0, 68.0],
                                                 [0.0, 33.0, 49.0, 39.0, 58.0, 67.0, 60.0, 54.0, 52.0, 52.0, 67.0, 69.0, 71.0],
                                                 [0.0, 33.0, 42.0, 39.0, 44.0, 52.0, 49.0, 42.0, 45.0, 46.0, 52.0, 53.0, 56.0]]  )

        capabilitiesMatrix = dtype(capabilitiesMatrix)

        observedTaskID = int(observedTaskID)

        observedCapability = torch.squeeze(capabilitiesMatrix[:, observedTaskID])

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
    
    def computeTrust(self, inptaskspredID):

        requiredCapability = self.requirementTransform(inptaskspredID)

        trust = 1.0

        for i in range(self.capabilityRepresentationSize):

            p_i = self.betas[i] * (requiredCapability[i] - self.capabilityMean[i])
            d_i = ( 1 + torch.exp(p_i) ) ** ( -self.zetas[i] )

            trust = trust * d_i

        print(self.zetas)

        return dtype(trust)