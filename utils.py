import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import pipeline
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm
import librosa
import random
import os
import sys
from dataset import CMU_MOSEIASRDataset, CMU_MOSEISLUDataset

def prepare_CMU_MOSEI_ASR_dataset(data_dir, num_class, preprocessor):
    train_data, dev_data, test_data = [], [], []
    df = pd.read_csv(data_dir + "/CMU_MOSEI_Labels.csv")
    for row in df.itertuples():
        filename = row.file + '_' + str(row.index) + '.wav'
        if num_class == 2:
            label = row.label2a
        else:
            # Avoid CUDA error: device-side assert triggered (due to negative label)
            label = row.label7 + 3
        if row.split == 0:
            train_data.append((filename, label))
        elif row.split == 1:
            dev_data.append((filename, label))
        elif row.split == 2:
            test_data.append((filename, label))

    return { "train": CMU_MOSEIASRDataset('train', train_data, data_dir, preprocessor),
            "dev": CMU_MOSEIASRDataset('dev', dev_data, data_dir, preprocessor),
            "test":  CMU_MOSEIASRDataset('test', test_data, data_dir, preprocessor)}

def prepare_CMU_MOSEI_SLU_dataset(data_dir, num_class):
    train_data, dev_data, test_data = [], [], []
    df = pd.read_csv(data_dir + "/CMU_MOSEI_Labels.csv")
    for row in df.itertuples():
        filename = row.file + '_' + str(row.index) + '.wav'
        if num_class == 2:
            label = row.label2a
        else:
            # Avoid CUDA error: device-side assert triggered (due to negative label)
            label = row.label7 + 3
        if row.split == 0:
            train_data.append((filename, label))
        elif row.split == 1:
            dev_data.append((filename, label))
        elif row.split == 2:
            test_data.append((filename, label))

    return { "train": CMU_MOSEISLUDataset('train', train_data, data_dir),
            "dev": CMU_MOSEISLUDataset('dev', dev_data, data_dir),
            "test":  CMU_MOSEISLUDataset('test', test_data, data_dir)}
