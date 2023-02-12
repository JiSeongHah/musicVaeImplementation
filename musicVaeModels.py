import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# 인코더 파트
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

# 컨덕터 파트
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

# 디코더 파트
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

# 위  세모델을 이용하여 메인 모델 생성

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

        # 레이턴트 벡터 크기
        self.FVSize = FVSize
        # 최종 아웃풋 크기
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

        # 인코더 직후 리니어 층, 이 층을 지난 결과를 둘로 나눠 mu, sigma로 사용
        self.linearEnc = nn.Linear(self.encHiddenSize *2 ,self.FVSize *2)
        # 논문에 언급된, 리니어 층
        self.linearBefZ = nn.Linear(self.FVSize,self.decInputSize)
        # 논문에 언급된 마지막 리니어 층
        self.linearFinal = nn.Linear(self.decInputSize,self.finalSize)

        self.dropOut = nn.Dropout(p=self.dropRate)



    def forward(self,x):

        pass

    # 인코더 및 인코더 직후 리니어층 만 통과
    def doEncode(self,x,h0,c0):

        out,_= self.encoder(x,h0,c0)

        out = self.linearEnc(out)

        mu,logVar = torch.chunk(out,2,dim=-1)

        logVar = F.softplus(logVar)

        return mu, logVar

    # 컨덕터 및 디코더를 포함한 기타 층 통과
    def doDecode(self,z,notes,conHidden,decMul,stepPerBar,doTeacherForcing,teachLabel=None):

        bSize = z.size(0)

        firstNote = torch.zeros(bSize, 1, self.finalSize, device=z.device)

        z = self.linearBefZ(z)

        z = torch.tanh(z)

        # 논문에선 스케쥴드 방식을 사용하나 티쳐 포싱도 언급 되어 있어, 시간상 4 bar임에도 티쳐 포싱 사용
        if doTeacherForcing:

            for i in range(4):
                # print(f'conHidden size : {conHidden[0].size()}')
                # print(f'input size: {z[:,16*i,:].view(bSize,1,-1).size()}')
                # 컨덕터 지난 후 결과
                conResult,_ = self.conductor(z[:,16*i,:].view(bSize,1,-1),conHidden)
                # print(f'first conResult size : {conResult.size()}')

                # 디코더에 들어가기 위한 초기 h0, c0
                decHidden = (torch.randn(1*decMul,
                                  bSize,
                                  self.decInputSize,
                                  device=z.device),

                              torch.randn(1*decMul,
                                    bSize,
                                    self.decInputSize,
                                    device=z.device))
                # print(f'decHidden size : {decHidden[0].size()}')

                # 디코더 인풋과 사이즈 맞추기 위해 expand
                conResult = conResult.expand(bSize,stepPerBar,conResult.size(2))
                # print(f'expanded conResult size : {conResult.size()}')
                # print(f'teacherLabel size : {teachLabel[:,16*i:16*(i+1),:].size()}')

                # 티쳐 포싱 할 데이터와 컨덕터 결과를 concat 해서 사용
                decInput = torch.cat([conResult,teachLabel[:,16*i:16*(i+1),:]],dim=-1)
                # print(f'decInput SIze : {decInput.size()}')

                # 디코더 인풋
                decOut,decHidden = self.decoder(decInput,decHidden)
                # print(f'decOut size : {decOut.size()}')

                # 디코더 지난 후 논문에서 언급된 최종 리니어 층 통과
                decOut = self.linearFinal(decOut)
                # decOut = torch.tanh(decOut)
                # print(f'decOut size : {decOut.size()}')
                # print('')

                # 최종 결과를 담기 위한 텐서
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
                # decOut = torch.tanh(decOut)

                notes[:,16*i:16*(i+1),:] = decOut


        return notes

    # 랜덤 분포에서 새로운 데이터 생성
    def genNew(self,genZ,h0,c0,generated,decMul):


        # fake Z 벡터
        genBSize = genZ.size(0)

        # 컨덕터에 들어간 초기 h0 c0
        conHidden = (torch.zeros(1 * self.conLayerNum, genBSize, self.decInputSize, device=genZ.device),
                     torch.zeros(1 * self.conLayerNum, genBSize, self.decInputSize, device=genZ.device))

        # 첫번째 입력
        firstNote = torch.zeros(genBSize, 1, self.finalSize, device=genZ.device)

        # 몇번쨰 bar에 집어넣을 것인지 counter
        cnt = 0
        # 결과를 담기위한 텐서
        generated = torch.zeros(genBSize, 64, self.finalSize, device=genZ.device)
        with torch.set_grad_enabled(False):

            # 4bar 이므로 4로 하드 코딩
            for i in range(4):

                # 디코더 초기 입력 h0,c0 논문에서 언급된 대로 매번 새로운 걸로 사용
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

                # 컨덕터 결과
                conResult, conHidden = self.conductor(genZ[:,16*i,:].view(genBSize,1,-1),conHidden)
                print(f'conResul size : {conResult.size()}')

                # 1 bar 당 step이 16이므로 16으로 하드 코딩
                for j in range(16):

                    # 컨덕터와 초기 입력을 컨켓
                    decInput = torch.cat([conResult, firstNote], dim=-1)
                    decInput = decInput.view(genBSize, 1, -1)
                    print(f'decInptut size: {decInput.size()}')

                    # 디코더 아웃풋
                    decOut, decHidden = self.decoder(decInput, decHidden)
                    decOut = self.linearFinal(decOut)

                    # 되먹이기 위해 초기 입력 텐서를 이전 결과로 치환
                    firstNote = decOut

                    # dim 2의 27 중 9 , 9, 9 가 각각 hit, velocity ,offset으로 정의되어 있음
                    outHit,outVelo,outOffset = torch.chunk(decOut,3,dim=-1)

                    # 히트 범위 : 바이너리
                    outHit = torch.sigmoid(outHit)
                    # 벨로시티 : 0~1 사이 연속값
                    outVelo = torch.sigmoid(outVelo)
                    # 오프셋 : -1~1 사이 연솟값
                    outOffset = torch.tanh(outOffset)
                    print(f'outHit.size() : {outHit.size()}')
                    # 다시 컨켓
                    decOut = torch.cat([outHit,outVelo,outOffset],dim=-1)
                    print(f'fdecOut size : {decOut.size()}')

                    print(f'generatee pard size : {generated[:, cnt, :].size()}')

                    #최종 결과
                    generated[:, cnt, :] = decOut.squeeze()


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
