import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence as KLD
from torch.distributions.normal import Normal

def betaElboLoss(pred,label,pMu,logVar,beta):

    likelihood = -F.binary_cross_entropy(pred,label,reduction='none')

    pSigma = torch.exp(logVar * 2)

    qMu = torch.tensor([0])
    qSigma = torch.tensor([1])

    distriP = Normal(pMu,pSigma)
    distriQ = Normal(qMu,qSigma)

    klDivergence = KLD(distriQ,distriP)

    betaELBO = torch.mean(likelihood) - beta * torch.mean(klDivergence)


    return -betaELBO










