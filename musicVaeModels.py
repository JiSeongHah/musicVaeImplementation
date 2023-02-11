import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class musicVaeEncoder(nn.Module):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 layerNum,
                 biDirectional=True
                 ):
        super(musicVaeEncoder,self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.layerNum = layerNum
        self.biDirectional = biDirectional

        self.encoder = torch.nn.LSTM(batch_first=True,
                                     input_size=self.inputSize,
                                     hidden_size=self.hiddenSize,
                                     num_layers=self.layerNum,
                                     bidirectional=biDirectional)

    def forward(self,x,h0,c0):

        out = self.encoder(x,(h0,c0))
        return out


class musicVaeConductor(nn.Module):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 layerNum,
                 biDirectional=False
                 ):
        super(musicVaeConductor,self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.layerNum = layerNum
        self.biDirectional = biDirectional

        self.conductor = torch.nn.LSTM(batch_first=True,
                                     input_size=self.inputSize,
                                     hidden_size=self.hiddenSize,
                                     num_layers=self.layerNum,
                                     bidirectional=biDirectional)

    def forward(self,x,conHidden):

        out = self.conductor(x,conHidden)
        return out


class musicVaeDecoder(nn.Module):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 layerNum,
                 biDirectional=False
                 ):
        super(musicVaeDecoder,self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.layerNum = layerNum
        self.biDirectional = biDirectional

        self.Decoder = torch.nn.LSTM(batch_first=True,
                                     input_size=self.inputSize,
                                     hidden_size=self.hiddenSize,
                                     num_layers=self.layerNum,
                                     bidirectional=biDirectional)

    def forward(self,x,decHidden):

        out = self.Decoder(x,decHidden)
        return out


class musicVaeMainModel(nn.Module):
    def __init__(self,
                 encInputSize,
                 encHiddenSize,
                 encLayerNum,
                 conInputSize,
                 conHiddenSize,
                 conLayerNum,
                 decInputSize,
                 decHiddenSize,
                 decLayerNum,
                 FVSize,
                 finalSize,
                 dropRate,
                 encBiDirectional=True,
                 conBiDirectional=False,
                 decBiDirectional=False
                 ):
        super(musicVaeMainModel,self).__init__()

        self.encInputSize = encInputSize
        self.encHiddenSize = encHiddenSize
        self.encLayerNum = encLayerNum

        self.conInputSize = conInputSize
        self.conHiddenSize = conHiddenSize
        self.conLayerNum = conLayerNum

        self.decInputSize = decInputSize
        self.decHiddenSize = decHiddenSize
        self.decLayerNum = decLayerNum

        self.encBiDirectional = encBiDirectional
        self.conBiDirectional = conBiDirectional
        self.decBiDirectional = decBiDirectional

        self.dropRate = dropRate

        self.FVSize = FVSize
        self.finalSize = finalSize

        self.encoder = musicVaeEncoder(inputSize=self.encInputSize,
                                       hiddenSize=self.encHiddenSize,
                                       layerNum=self.encLayerNum)

        self.conductor = musicVaeConductor(inputSize=self.conInputSize,
                                           hiddenSize=self.conHiddenSize,
                                           layerNum=self.conLayerNum)

        self.decoder = musicVaeDecoder(inputSize=self.decInputSize+self.finalSize,
                                       hiddenSize=self.decHiddenSize,
                                       layerNum=self.decLayerNum)

        self.linearEnc = nn.Linear(self.encHiddenSize *2 ,self.FVSize *2)
        self.linearBefZ = nn.Linear(self.FVSize,self.decInputSize)
        self.linearFinal = nn.Linear(self.decInputSize,self.finalSize)

        self.dropOut = nn.Dropout(p=self.dropRate)



    def forward(self,x):

        encH0, encC0, conH0,conC0 = self.makeFirstHidden()

        out = self.Decoder(x)
        return out

    def doEncode(self,x,h0,c0):

        out,_= self.encoder(x,h0,c0)

        out = self.linearEnc(out)

        mu,logVar = torch.chunk(out,2,dim=-1)

        logVar = F.softplus(logVar)

        return mu, logVar

    def doDecode(self,z,notes,conHidden,decMul,stepPerBar,doTeacherForcing,teachLabel=None):

        bSize = z.size(0)

        firstNote = torch.zeros(bSize, 1, self.finalSize, device=z.device)

        z = self.linearBefZ(z)

        z = torch.tanh(z)

        if doTeacherForcing:

            for i in range(4):
                # print(f'conHidden size : {conHidden[0].size()}')
                # print(f'input size: {z[:,16*i,:].view(bSize,1,-1).size()}')
                conResult,_ = self.conductor(z[:,16*i,:].view(bSize,1,-1),conHidden)

                decHidden = (torch.randn(1*decMul,
                                  bSize,
                                  self.decInputSize,
                                  device=z.device),

                              torch.randn(1*decMul,
                                    bSize,
                                    self.decInputSize,
                                    device=z.device))

                conResult = conResult.expand(bSize,stepPerBar,conResult.size(2))

                decInput = torch.cat([conResult,teachLabel[:,16*i:16*(i+1),:]],dim=-1)

                decOut,decHidden = self.decoder(decInput,decHidden)

                decOut = self.linearFinal(decOut)
                decOut = torch.softmax(decOut,dim=2)
                print(f'decOut size : {decOut.size()}')

                notes[:,16*i:16*(i+1),:] = decOut

        else:

            for i in range(4):

                conResult,_ = self.conductor(z[:,16*i,:].view(bSize,1,-1),conHidden)

                decHidden = (torch.randn(1*decMul,
                                  bSize,
                                  self.decInputSize,
                                  device=z.device),

                              torch.randn(1*decMul,
                                    bSize,
                                    self.decInputSize,
                                    device=z.device))



                decInput = torch.cat([conResult,firstNote],dim=-1)
                decInput = decInput.view(bSize,1,-1)

                decOut,decHidden = self.decoder(decInput,decHidden)

                decOut = self.linearFinal(decOut)
                decOut = torch.sigmoid(decOut)

                notes[:,16*i:16*(i+1),:] = decOut


        return notes


    def genNew(self,genZ,h0,c0,generated,decMul):



        genBSize = genZ.size(0)

        conHidden = (torch.zeros(1 * self.conLayerNum, genBSize, self.decInputSize, device=genZ.device),
                     torch.zeros(1 * self.conLayerNum, genBSize, self.decInputSize, device=genZ.device))

        firstNote = torch.zeros(genBSize, 1, self.finalSize, device=genZ.device)

        cnt = 0
        generated = torch.zeros(genBSize, 64, self.finalSize, device=genZ.device)
        with torch.set_grad_enabled(False):

            for i in range(4):

                decHidden = (torch.randn(1*decMul,
                                         genBSize,
                                         self.decInputSize,
                                         device=genZ.device),

                             torch.randn(1*decMul,
                                         genBSize,
                                         self.decInputSize,
                                         device=genZ.device))

                print(f'genZ is on device: {genZ.device}')
                print(f'conHidden is on device : {conHidden[0].device}')

                conResult, conHidden = self.conductor(genZ[:,16*i,:].view(genBSize,1,-1),conHidden)

                for j in range(16):

                    decInput = torch.cat([conResult, firstNote], dim=-1)
                    decInput = decInput.view(genBSize, 1, -1)
                    print(f'decInptut size: {decInput.size()}')

                    decOut, decHidden = self.decoder(decInput, decHidden)

                    decOut = self.linearFinal(decOut)
                    decOut = torch.sigmoid(decOut)
                    print(f'fdecOut size : {decOut.size()}')

                    print(f'generatee pard size : {generated[:, cnt, :].size()}')

                    generated[:, cnt, :] = decOut.squeeze()

                    firstNote = decOut
                    print(f'cnt : {cnt}')
                    cnt +=1


        return generated
















#
#
#
# model = nn.LSTM(batch_first=True,bidirectional=True,input_size=13,hidden_size=2048)
# lin = nn.Linear(4096,512*2)
# x = torch.randn(32,64,13)
# h0 = torch.randn(2,32,2048)
# c0 = torch.randn(2,32,2048)
#
# y,_ = model(x,(h0,c0))
#
# # print(y)
# print(y.size())
# z = lin(y)
# print(z.size())
#
# mu,sig = torch.chunk(z,2,dim=-1)
#
# print(mu.size())
# print(sig.size())
#
# eps = torch.randn(32,1,512)
# print(eps.size())
# zz = mu+sig*eps
#
# print(zz.size())
# print(zz[:,16*0,:].size())
# print(zz[:,16*0,:].view(32,1,-1).size())
