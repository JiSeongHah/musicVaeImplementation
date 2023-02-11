import torch
import torch.nn as nn
from musicVaeModels import *
from musicVaeDataloading import *
from musicVaeLossFunc import betaElboLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from usefulFuncs import createDir

class musicVaeMainloop(nn.Module):

    def __init__(self,
                 dataLoadDir,
                 modelSaveLoadDir,
                 plotSaveDir,
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
                 trnBSize,
                 valBSize,
                 lr,
                 wDecay,
                 modelLoadNum,
                 beta,
                 dropRate=0.2,
                 stepPerBar = 16,
                 encBiDirectional=True,
                 conBiDirectional=False,
                 decBiDirectional=False,
                 useGpu = True):

        super(musicVaeMainloop,self).__init__()

        self.dataLoadDir = dataLoadDir
        self.modelSaveLoadDir = modelSaveLoadDir
        createDir(self.modelSaveLoadDir)
        self.plotSaveDir = plotSaveDir
        createDir(self.plotSaveDir)

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


        self.trnBSize=  trnBSize
        self.valBSize = valBSize
        self.lr = lr
        self.wDecay = wDecay
        self.beta = beta

        self.FVSize = FVSize
        self.finalSize = finalSize
        self.stepPerBar = stepPerBar
        self.totalStep = self.stepPerBar * 4
        self.modelLoadNum = modelLoadNum
        self.dropRate = dropRate

        self.model = musicVaeMainModel(encInputSize = self.encInputSize,
                                       encHiddenSize = self.encHiddenSize,
                                       encLayerNum = self.encLayerNum,
                                       conInputSize = self.conInputSize,
                                       conHiddenSize = self.conHiddenSize,
                                       conLayerNum = self.conLayerNum,
                                       decInputSize = self.decInputSize,
                                       decHiddenSize = self.decHiddenSize,
                                       decLayerNum = self.decLayerNum,
                                       encBiDirectional = self.encBiDirectional,
                                       conBiDirectional = self.conBiDirectional,
                                       decBiDirectional =self.decBiDirectional,
                                       FVSize=self.FVSize,
                                       finalSize=self.finalSize,
                                       dropRate=self.dropRate)


        self.optimizer = Adam(self.model.parameters(),
                              lr=self.lr,  # 학습률
                              eps=1e-9,
                              weight_decay=self.wDecay
                              )



        self.useGpu = useGpu
        ##### use GPU or not ###########################
        if self.useGpu == True:
            cudaPossible = torch.cuda.is_available()
            print('cuda status : ', cudaPossible)
            self.device = torch.device('cuda' if cudaPossible else 'cpu')
            print('trainig with device :', self.device)
        else:
            self.device = torch.device('cpu')
            print('trainig with device :', self.device)
        ##### use GPU or not ###########################


        self.trnDataset = musicVaeDataset(baseDir=self.dataLoadDir,
                                          isTrain=True)

        self.valDataset = musicVaeDataset(baseDir=self.dataLoadDir,
                                          isTrain=False)

        self.trnLossLst = []
        self.trnLossLstAvg = []

        self.valLossLst = []
        self.valLossLstAvg = []


    def forward(self,x):

        out= self.model(x)
        return out

    def calLoss(self,pred,label,pMu,logVar,beta):

        return betaElboLoss(pred,label,pMu,logVar,beta)

    def trainingStep(self):


        self.model.to(self.device)
        self.model.train()

        trainLoader = DataLoader(self.trnDataset,batch_size=self.trnBSize,shuffle=True)

        self.optimizer.zero_grad()
        for eachInput in tqdm(trainLoader):
            self.optimizer.zero_grad()

            bInput = eachInput['Input']
            bOutput = eachInput['Output']

            bInput = bInput.float().to(self.device)
            bSize = bInput.size(0)

            h0 = torch.zeros(2*self.encLayerNum,
                             bSize,
                             self.encHiddenSize,
                             device=self.device)

            c0 = torch.zeros(2*self.encLayerNum,
                             bSize,
                             self.encHiddenSize,
                             device=self.device)

            mu,logVar = self.model.doEncode(bInput,h0,c0)

            # print(mu.size(),logVar.size(),111)

            with torch.set_grad_enabled(False):
                epsilon = torch.randn(bSize,1,self.FVSize,device=self.device)

            sigma = torch.exp(logVar * 2)
            z = mu + sigma*epsilon

            # print(f'z size : {z.size()}')

            conHidden = (torch.zeros(1*self.conLayerNum,bSize,self.decInputSize,device=self.device),
                         torch.zeros(1*self.conLayerNum,bSize,self.decInputSize,device=self.device))

            # print(f'conHidden size : {conHidden[0].size()}')

            notes = torch.zeros(bSize,self.totalStep,self.finalSize,device=self.device)

            firstNote = torch.zeros(bSize, 1, self.finalSize, device=z.device)
            teacherLabel = torch.cat([firstNote,bInput.clone().detach()],dim=1)
            finalOut = self.model.doDecode(teachLabel=teacherLabel,
                                           z=z,
                                           notes=notes,
                                           conHidden=conHidden,
                                           decMul=self.decLayerNum,
                                           stepPerBar=self.stepPerBar,
                                           doTeacherForcing=True)
            finalOut = finalOut.cpu()
            # print(f'final output size : {finalOut.size()}')
            # print(f'final output is on device : {finalOut.device}')
            mu = mu.cpu()
            logVar = logVar.cpu()
            loss = betaElboLoss(pred=finalOut,
                                label=bOutput.float(),
                                pMu=mu,
                                logVar=logVar,
                                beta=self.beta)

            loss.backward()
            self.optimizer.step()
            self.trnLossLst.append(loss.item())

        self.model.to('cpu')
        self.model.eval()

    def trainingStepEnd(self):

        self.trnLossLstAvg.append(np.mean(self.trnLossLst))

        plt.plot(range(len(self.trnLossLstAvg)),self.trnLossLstAvg)
        plt.savefig(os.path.join(self.plotSaveDir,'trnLoss.png'),dpi=200)
        plt.close()
        plt.cla()
        plt.clf()

    def doTrain(self):
        self.trainingStep()
        self.trainingStepEnd()

    def validationStep(self):


        self.model.to(self.device)
        self.model.eval()

        trainLoader = DataLoader(self.valDataset,batch_size=self.valBSize,shuffle=False)

        self.optimizer.zero_grad()
        for eachInput in tqdm(trainLoader):


            bInput = eachInput['Input']
            bOutput = eachInput['Output']

            bInput = bInput.float().to(self.device)
            bSize = bInput.size(0)

            h0 = torch.zeros(2*self.encLayerNum,
                             bSize,
                             self.encHiddenSize,
                             device=self.device)

            c0 = torch.zeros(2*self.encLayerNum,
                             bSize,
                             self.encHiddenSize,
                             device=self.device)

            mu,logVar = self.model.doEncode(bInput,h0,c0)

            # print(mu.size(),logVar.size(),111)

            with torch.set_grad_enabled(False):
                epsilon = torch.randn(bSize,1,self.FVSize,device=self.device)

            sigma = torch.exp(logVar * 2)
            z = mu + sigma*epsilon

            # print(f'z size : {z.size()}')

            conHidden = (torch.zeros(1*self.conLayerNum,bSize,self.decInputSize,device=self.device),
                         torch.zeros(1*self.conLayerNum,bSize,self.decInputSize,device=self.device))

            # print(f'conHidden size : {conHidden[0].size()}')

            notes = torch.zeros(bSize,self.totalStep,self.finalSize,device=self.device)

            firstNote = torch.zeros(bSize, 1, self.finalSize, device=z.device)
            teacherLabel = torch.cat([firstNote,bInput.clone().detach()],dim=1)
            finalOut = self.model.doDecode(teachLabel=teacherLabel,
                                           z=z,
                                           notes=notes,
                                           conHidden=conHidden,
                                           decMul=self.decLayerNum,
                                           stepPerBar=self.stepPerBar,
                                           doTeacherForcing=False)
            finalOut = finalOut.cpu()
            # print(f'final output size : {finalOut.size()}')
            # print(f'final output is on device : {finalOut.device}')
            mu = mu.cpu()
            logVar = logVar.cpu()
            loss = betaElboLoss(pred=finalOut,
                                label=bOutput.float(),
                                pMu=mu,
                                logVar=logVar,
                                beta=self.beta)


            self.valLossLst.append(loss.item())

        self.model.to('cpu')

    def validationStepEnd(self):

        self.valLossLstAvg.append(np.mean(self.valLossLst))

        plt.plot(range(len(self.valLossLstAvg)), self.valLossLstAvg)
        plt.savefig(os.path.join(self.plotSaveDir, 'valLoss.png'), dpi=200)
        plt.close()
        plt.cla()
        plt.clf()

    def doVal(self):

        self.validationStep()
        self.validationStepEnd()

    def doTrainVal(self):

        self.doTrain()
        self.doVal()

    def genNew(self,genBSize):

        self.model.to(self.device)
        self.model.eval()

        genZ = torch.randn(genBSize,self.totalStep,self.conHiddenSize,device=self.device)

        h0 = torch.zeros(2 * self.encLayerNum,
                         genBSize,
                         self.encHiddenSize,
                         device=self.device)

        c0 = torch.zeros(2 * self.encLayerNum,
                         genBSize,
                         self.encHiddenSize,
                         device=self.device)



        generated = torch.zeros(genBSize,self.totalStep,self.finalSize,device=self.device)


        generated = self.model.genNew(genZ=genZ,
                                      h0=h0,
                                      c0=c0,
                                      generated=generated,
                                      decMul=self.decLayerNum)

        self.model.to('cpu')
        return generated.cpu()


    def genInterpol(self):

        self.model.to(self.device)
        self.model.eval()

        trainLoader = DataLoader(self.valDataset, batch_size=7, shuffle=False)

        self.optimizer.zero_grad()
        for eachInput in tqdm(trainLoader):
            bInput = eachInput['Input']
            bOutput = eachInput['Output']

            bInput = bInput.float().to(self.device)
            bSize = bInput.size(0)

            h0 = torch.zeros(2 * self.encLayerNum,
                             bSize,
                             self.encHiddenSize,
                             device=self.device)

            c0 = torch.zeros(2 * self.encLayerNum,
                             bSize,
                             self.encHiddenSize,
                             device=self.device)

            mu, logVar = self.model.doEncode(bInput, h0, c0)

            # print(mu.size(),logVar.size(),111)

            with torch.set_grad_enabled(False):
                epsilon = torch.randn(bSize, 1, self.FVSize, device=self.device)

            sigma = torch.exp(logVar * 2)
            z = mu + sigma * epsilon

            break

        genA,genB= torch.chunk(z,2,dim=-1)
        print(genA.size(),genB.size())
        genZ = genA+genB
        genBSize = genZ.size(0)




        # genZ = torch.randn(genBSize,self.totalStep,self.conHiddenSize,device=self.device)

        h0 = torch.zeros(2 * self.encLayerNum,
                         genBSize,
                         self.encHiddenSize,
                         device=self.device)

        c0 = torch.zeros(2 * self.encLayerNum,
                         genBSize,
                         self.encHiddenSize,
                         device=self.device)



        generated = torch.zeros(genBSize,self.totalStep,self.finalSize,device=self.device)


        generated = self.model.genNew(genZ=genZ,
                                      h0=h0,
                                      c0=c0,
                                      generated=generated,
                                      decMul=self.decLayerNum)

        self.model.to('cpu')
        return generated.cpu()































