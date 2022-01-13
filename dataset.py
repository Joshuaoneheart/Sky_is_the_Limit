import random
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

import os
import torchaudio

'''
SAMPLE_RATE = 16000
EXAMPLE_WAV_MIN_SEC = 5
EXAMPLE_WAV_MAX_SEC = 15
EXAMPLE_DATASET_SIZE = 10000
'''
class CMU_MOSEIASRDataset(Dataset):
    def __init__(self, split, data, path, processor):
        self.split = split
        self.data = data
        self.path = path
        self.processor = processor

    def __getitem__(self, idx):
        wav_path = os.path.join(self.path, 'Audio', self.split, self.data[idx][0])
        wav, sr = torchaudio.load(wav_path)
        text_path = os.path.join(self.path, 'Transcript/Combined', "_".join(self.data[idx][0].split("_")[:-1]) + ".txt")
        text = ""
        with open(text_path, "r") as f:
            for line in f.readlines():
                if line.split("___")[1] == self.data[idx][0].split("_")[-1].replace(".wav", ""):
                    text = line.split("___")[-1]

        item = {}
        item["input_values"] = self.processor(wav, sampling_rate=sr, return_tensors="pt").input_values[0]
        item["input_length"] = len(item["input_values"])
        
        
        '''
        wav_sec = random.randint(EXAMPLE_WAV_MIN_SEC, EXAMPLE_WAV_MAX_SEC)
        wav = torch.randn(SAMPLE_RATE * wav_sec)
        label = random.randint(0, self.class_num - 1)
        '''
        
        with self.processor.as_target_processor():
            item["labels"] = self.processor(text, return_tensors="pt").input_ids

        return item

    def __len__(self):
        return len(self.data)

class CMU_MOSEISLUDataset(Dataset):
    def __init__(self, split, data, path):
        self.split = split
        self.data = data
        self.path = path

    def __getitem__(self, idx):
        wav_path = os.path.join(self.path, 'Audio', self.split, self.data[idx][0])
        wav, sr = torchaudio.load(wav_path)
        max_l = 800000
        if wav.shape[1] > max_l:
            wav = wav[:, :max_l]
        if wav.shape[1] < 32000:
            wav = nn.functional.pad(wav, (0, 32000 - wav.shape[1]))

        label = self.data[idx][1]

        '''
        wav_sec = random.randint(EXAMPLE_WAV_MIN_SEC, EXAMPLE_WAV_MAX_SEC)
        wav = torch.randn(SAMPLE_RATE * wav_sec)
        label = random.randint(0, self.class_num - 1)
        '''

        return wav.view(-1), torch.tensor(label).long()

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
