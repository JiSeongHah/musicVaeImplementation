import torch
import torch.nn as nn
from musicVaeMainLoop import musicVaeMainloop
import magenta.models.music_vae.data as mVaeData
import magenta.models.music_vae as mVae
import os

dataLoadDir = '/home/a286winteriscoming/Downloads/groove-v1.0.0/groove/'
baseDir = '/home/a286winteriscoming/musicVaeTest1/'
modelSaveLoadDir = os.path.join(baseDir,'modelSaveLoad')
plotSaveDir = os.path.join(baseDir,'plots')

encInputSize = 27
encHiddenSize = 512
encLayerNum = 2

decInputSize = 256
decHiddenSize = 256
decLayerNum=2

conInputSize = 256
conHiddenSize = 256
conLayerNum = 2



FVSize = 512
finalSize = 27
trnBSize = 256
valBSize = 1
lr = 1e-3
wDecay = 0.9999
modelLoadNum = 1
beta = 0.2


loop = musicVaeMainloop(dataLoadDir = dataLoadDir,

                        modelSaveLoadDir = modelSaveLoadDir,
                        plotSaveDir = plotSaveDir,

                        encInputSize = encInputSize,
                        encHiddenSize = encHiddenSize,
                        encLayerNum = encLayerNum,

                        decInputSize = decInputSize,
                        decHiddenSize = decHiddenSize,
                        decLayerNum = decLayerNum,

                        conInputSize = conInputSize,
                        conHiddenSize = conHiddenSize,
                        conLayerNum = conLayerNum,

                        FVSize = FVSize,
                        finalSize = finalSize,
                        trnBSize = trnBSize,
                        valBSize = valBSize,
                        lr = lr,
                        wDecay = wDecay,
                        modelLoadNum = modelLoadNum,
                        beta = beta)

for i in range(1):
    loop.doTrainVal()


gened = loop.genInterpol()

print(gened.size())

converter = mVae.data.GrooveConverter(split_bars=4,
                                              steps_per_quarter=4,
                                              quarters_per_bar=4,
                                              max_tensors_per_notesequence=20,
                                              pitch_classes=mVaeData.ROLAND_DRUM_PITCH_CLASSES,
                                              inference_pitch_classes=mVaeData.REDUCED_DRUM_PITCH_CLASSES)

lst = []
for i in gened:
    lst.append(i.cpu().numpy())

newNoteSeq = converter.from_tensors(lst)
print(newNoteSeq)
