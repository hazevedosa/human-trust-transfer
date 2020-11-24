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

dtype = torch.DoubleTensor

if usecuda:
    dtype = torch.cuda.FloatTensor



class RobotTrustModel(torch.nn.Module):

    def __init__(self):
        super(RobotTrustModel, self).__init__()

        # self.lambda_l = Parameter(dtype(np.zeros(1)))
        # self.lambda_u = Parameter(dtype(np.ones(1)))
        # self.beta = Parameter(dtype(20.0 * np.random.rand(1)))
        # self.beta = dtype([1000.0])

        self.pre_beta = Parameter(dtype(4.0 * np.ones(1)))
        self.pre_lambda_l = Parameter(dtype(-10.0 * np.ones(1)))
        self.pre_lambda_u = Parameter(dtype( 10.0 * np.ones(1)))


    def forward(self, bin_centers):

        n_bins = bin_centers.shape[0]
        trust = torch.zeros(n_bins)

        lambda_l = self.sigm(self.pre_lambda_l)
        lambda_u = self.sigm(self.pre_lambda_u)
        beta = self.pre_beta * self.pre_beta

        # print(lambda_l)
        # print(lambda_u)
        # print(beta)

        # stop()


        for i in range(n_bins):
            trust[i] = self.compute_trust(lambda_l, lambda_u, beta, bin_centers[i])


        return trust.cuda()

    def compute_trust(self, l, u, b, p):

        if b < -50:
            trust = 1.0 - 1.0 / (b * (u - l)) * torch.log( (1.0 + torch.exp(b * (p - l))) / (1.0 + torch.exp(b * (p - u))) )
        else:
            if p <= l:
                trust = torch.tensor([1.0])
            elif p > u:
                trust = torch.tensor([0.0])
            else:
                trust = (u - p) / (u - l + 0.0001)

        return trust

    def sigm(self, x):
        return 1 / (1 + torch.exp(-x))




if __name__ == "__main__":
    

    model = RobotTrustModel()
    model.cuda()

    bin_c = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    bin_c = dtype(bin_c)

    obs_probs = [1, 1, 1, 0.8667, 0.2500, 0.1429, 0, 0, 0, 0]
    obs_probs = dtype(obs_probs)

    learning_rate = 0.1
    weight_decay = 0.01

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    t = 0

    report_period = 100

    while t < 500:

        def closure():
            diff = model(bin_c) - obs_probs
            loss = torch.mean( torch.pow(diff, 2.0) )
            optimizer.zero_grad()

            loss.backward()

            return loss

        optimizer.step(closure)

        if t % report_period == 0:
            print("\nt =", t)
            print("\n\nbeta =", model.pre_beta * model.pre_beta)
            print("lambda_l =", model.sigm(model.pre_lambda_l))
            print("lambda_u =", model.sigm(model.pre_lambda_u))

            loss_to_print = torch.mean( torch.pow( (model(bin_c) - obs_probs), 2.0 ) )
            print("\nloss", loss_to_print)

        t = t + 1

