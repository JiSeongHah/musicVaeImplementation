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







# 모든 mid 파일을 불러와서 (N,64,27) 형태로 저장하여 사용
# eval_session 폴더 내에 있는 것들은 validation으로 사용
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


        # 라이브러리의 컨버터를 이용해 미디 파일을 넘파이 어레이로 변경
        converter = mVae.data.GrooveConverter(split_bars=4,
                                              steps_per_quarter=4,
                                              quarters_per_bar=4,
                                              max_tensors_per_notesequence=20,
                                              pitch_classes=mVaeData.ROLAND_DRUM_PITCH_CLASSES,
                                              inference_pitch_classes=mVaeData.REDUCED_DRUM_PITCH_CLASSES)

        inputTensor = []
        outputTensor = []

        # 각 미디 파일을 열어서 어레이를 리스트에 저장
        for eachPath in self.dataPathLst:
            loadedNoteSeq = note_seq.midi_io.midi_file_to_note_sequence(eachPath)

            tensored = converter.to_tensors(loadedNoteSeq)

            eachInputLst = tensored.inputs
            eachOutputLst = tensored.outputs

            for eachInput,eachOutput in zip(eachInputLst,eachOutputLst):
                inputTensor.append(torch.from_numpy(eachInput))
                outputTensor.append(torch.from_numpy(eachOutput))

        # 저장이 끝난 리스트를 다시 파이토치 텐서로 변환
        self.inputTensor = torch.stack(inputTensor)
        self.outputTensor = torch.stack(outputTensor)


    def __len__(self):

        return len(self.inputTensor)

    def __getitem__(self, idx):

        Input = self.inputTensor[idx]
        Output = self.outputTensor[idx]

        totalData = {'Input':Input,'Output':Output}

        return totalData

