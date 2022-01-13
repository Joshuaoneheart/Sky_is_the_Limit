import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AdamW, Wav2Vec2ForSequenceClassification, Wav2Vec2Config
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
import librosa
import random
import os
from utils import prepare_CMU_MOSEI_SLU_dataset
import sys
config = {
    "train_batch_size": 1,
    "logging_step": 1000,
    "padding_length": 32000,
    "max_length": 300000,
    "sample_rate": 16000,
    "lr": 1e-5,
    "num_class": 2
}

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        configuration = Wav2Vec2Config(
            num_labels=config["num_class"], use_weighted_layer_sum=True, classifier_proj_size=32)
        self.w2v = Wav2Vec2ForSequenceClassification(configuration)
        self.w2v.wav2vec2.from_pretrained("facebook/wav2vec2-base-960h")

    def forward(self, x):
        x = self.w2v(input_values=x)
        #print(f'x = {x}')
        return x.logits


"""### Training"""

datasets = prepare_CMU_MOSEI_SLU_dataset("data/CMU_MOSEI", config["num_class"])
train_loader = DataLoader(
    datasets["train"], batch_size=config["train_batch_size"], shuffle=True, drop_last=False)
valid_loader = DataLoader(datasets["dev"], batch_size=1,
                          shuffle=False, drop_last=False)
test_loader = DataLoader(datasets["test"], batch_size=1,
                         shuffle=False, drop_last=False)

###
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"running on {device}")

model = Classifier().to(device)
model.to(device)
optimizer = AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.98))
try:
    checkpoint = torch.load("w2v_cmu.ckpt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Successfully load model")
except:
    pass

criterion = nn.CrossEntropyLoss()


n_epochs = 30
accu_step = 1
best_acc = 0
for epoch in range(n_epochs):
    model.train()
    train_loss = []
    train_accs = []
    step = 0
    for batch in tqdm(train_loader, file=sys.stdout):
        wavs, labels = batch
        wavs = torch.squeeze(wavs, 1).to(device)
        logits = model(wavs)

        loss = criterion(logits, labels.to(device))
        train_loss.append(loss.item())
        loss /= accu_step
        loss.backward()
        step += 1
        if step % accu_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        acc = (logits.argmax(dim=-1).cpu() == labels.cpu()).float().mean()

        train_accs.append(acc)
        if(step % (config["logging_step"] / config["train_batch_size"]) == 0):
            print(f"Loss: {sum(train_loss) / len(train_loss)}")
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    print(
        f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    model.eval()
    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader, file=sys.stdout):
        wavs, labels = batch
        with torch.no_grad():
            logits = model(wavs.to(device))
        loss = criterion(logits, labels.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    if valid_acc >= best_acc:
        best_acc = valid_acc
        print(f"Save model with acc {best_acc}")
        torch.save({"model": model.state_dict(),
                   "optimizer": optimizer.state_dict()}, "w2v_cmu.ckpt")

    print(
        f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
try:
    checkpoint = torch.load("w2v_cmu.ckpt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Successfully load model")
except:
    pass

# Testing
test_loss = []
test_accs = []
for batch in tqdm(test_loader, file=sys.stdout):
    wavs, labels = batch
    with torch.no_grad():
        logits = model(wavs.to(device))
    loss = criterion(logits, labels.to(device))
    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    test_loss.append(loss.item())
    test_accs.append(acc)
test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_accs) / len(test_accs)
print(f"[ Test | loss = {test_loss:.5f}, acc = {test_acc:.5f}")
