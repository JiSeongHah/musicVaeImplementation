import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence as KLD
from torch.distributions.normal import Normal

# beta elbo loss 구현
# 구현 잘못인지 hitloss 앞에 - 를 붙여야 제대로 생성이 됨을 확인
# 실제로는 안 붙이는 게 수학적으로 맞음
def betaElboLoss(pred,label,pMu,logVar,beta):

    # value of data tends to go down to -1 or up to thsu mse loss is correct one
    # furthermore library also uses mse loss

    predHit,predVelo,predOffset = torch.chunk(pred,3,dim=-1)
    labelHit,labelVelo,labelOffset = torch.chunk(label,3,dim=-1)

    predHit = torch.sigmoid(predHit)
    predVelo = torch.sigmoid(predVelo)
    predOffset = torch.tanh(predOffset)

    hitLoss = -F.binary_cross_entropy(predHit,labelHit)
    veloLoss = F.mse_loss(predVelo,labelVelo)
    offsetLoss = F.mse_loss(predOffset,labelOffset)

    # likelihood = -(hitLoss+veloLoss+offsetLoss)


    pSigma = torch.exp(logVar * 2)

    # qMu = torch.tensor([0])
    # qSigma = torch.tensor([1])
    #
    # distriP = Normal(pMu,pSigma)
    # distriQ = Normal(qMu,qSigma)
    #
    # klDivergence = KLD(distriQ,distriP)

    klDivergence = torch.sum(1 + logVar - pMu**2 - torch.exp(logVar), dim=(1,2))
    # log2 = torch.tensor([2.0])

    # freeNat = 48.0 * torch.log(log2)
    #
    # klLoss = torch.maximum(klDivergence-freeNat,0)

    # betaELBO = torch.mean(likelihood)

    return hitLoss+veloLoss+offsetLoss- beta * torch.mean(klDivergence)










