import note_seq.midi_io
import torch
import torch.nn as nn
import numpy as np
import magenta.models.music_vae.data as mVaeData
import magenta.models.music_vae as mVae
import note_seq
import tensorflow
import tensorflow_datasets as tfds
import os
from torch.utils.data import Dataset
from findMidiFles import getAllMidiFles,splitTrainVal

testDir = '/home/a286winteriscoming/Downloads/groove-v1.0.0/groove/drummer1/session1/1_funk_80_beat_4-4.mid'
# testDir = '/home/a286winteriscoming/Downloads/groove-v1.0.0/groove/'
#
#
# lst = getAllMidiFles(testDir)
# print(len(lst))
# trnLst,valLst = splitTrainVal(lst)
# print(len(trnLst),len(valLst))

converter = mVae.data.GrooveConverter(split_bars=4,
                                              steps_per_quarter=4,
                                              quarters_per_bar=4,
                                              max_tensors_per_notesequence=20,
                                              pitch_classes=mVaeData.ROLAND_DRUM_PITCH_CLASSES,
                                              inference_pitch_classes=mVaeData.REDUCED_DRUM_PITCH_CLASSES)
loadedNoteSeq = note_seq.midi_io.midi_file_to_note_sequence(testDir)
bef = converter.to_tensors(loadedNoteSeq)
#
for i in bef.inputs:
    for j in i:
        print(j)


x = torch.randn(2,1,3)
print(x)
y= torch.softmax(x,dim=2)
print(y)
#
# aft = converter.from_tensors(bef.inputs)
#
# print(aft)







#
class musicVaeDataset(Dataset):
    def __init__(self,
                 baseDir,
                 isTrain):
        super(musicVaeDataset,self).__init__()

        self.baseDir = baseDir
        self.isTrain = isTrain

        self.rawLst = getAllMidiFles(self.baseDir)

        self.trnPathLst,self.valPathLst = splitTrainVal(self.rawLst)

        if self.isTrain:
            self.dataPathLst = self.trnPathLst
        else:
            self.dataPathLst = self.valPathLst

        converter = mVae.data.GrooveConverter(split_bars=4,
                                              steps_per_quarter=4,
                                              quarters_per_bar=4,
                                              max_tensors_per_notesequence=20,
                                              pitch_classes=mVaeData.ROLAND_DRUM_PITCH_CLASSES,
                                              inference_pitch_classes=mVaeData.REDUCED_DRUM_PITCH_CLASSES)

        inputTensor = []
        outputTensor = []
        for eachPath in self.dataPathLst:
            loadedNoteSeq = note_seq.midi_io.midi_file_to_note_sequence(eachPath)

            tensored = converter.to_tensors(loadedNoteSeq)

            eachInputLst = tensored.inputs
            eachOutputLst = tensored.outputs

            for eachInput,eachOutput in zip(eachInputLst,eachOutputLst):
                inputTensor.append(torch.from_numpy(eachInput))
                outputTensor.append(torch.from_numpy(eachOutput))

        self.inputTensor = torch.stack(inputTensor)
        self.outputTensor = torch.stack(outputTensor)


    def __len__(self):

        return len(self.inputTensor)

    def __getitem__(self, idx):

        Input = self.inputTensor[idx]
        Output = self.outputTensor[idx]

        totalData = {'Input':Input,'Output':Output}

        return totalData

