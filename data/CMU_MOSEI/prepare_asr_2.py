import os
import fire
import numpy as np
import pandas as pd
import soundfile
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("torch_ckpt")
model.eval()

device = "cuda:0"
model.to(device)



from multiprocessing import Pool
from tqdm.auto import tqdm

split2kaldi = {
    "train": "fine-tune",
    "dev": "dev",
    "test": "test",
}

split2idx = {
    "train": 0,
    "dev": 1,
    "test": 2,
}

splits = {"test"}

def create_manifest(
    data_dir="",
    manifest_dir="manifest_asr_2"
):
    os.makedirs(manifest_dir, exist_ok=True)

    labels = sorted([0,1])
    with open(os.path.join(manifest_dir, f"labels.sent.txt"), "w") as f:
        for l in labels:
            print(l, file=f)
    df = pd.read_csv(os.path.join(data_dir, f"CMU_MOSEI_Labels.csv"))
    #'Unnamed: 0', 'file', 'index', 'start', 'end', 'label2a', 'label2b', 'label7', 'split'
    print(df.iloc[0])
    for split in splits:
        split_dir = os.path.join(data_dir, "Audio", split)
        os.makedirs(split_dir, exist_ok=True)

        with open(os.path.join(manifest_dir, f"{split2kaldi[split]}.tsv"), "w") as fp:
            print(os.path.abspath(os.path.join(data_dir, "Audio", split)), file=fp)
            for f, idx, start, end, label, sp in zip(
                df["file"].array, df["index"].array, df["start"].array, df["end"].array, df["label2a"].array, df["split"].array
            ):
                if(split2idx[split] != sp): continue
                frames = int(16000 * (end - start))
                print(f"{f}_{idx}.wav\t{frames}", file=fp)
                text = ""
                text_path = os.path.join(data_dir, f"Transcript/Combined/{f}.txt")
                audio, sr = soundfile.read(f"Audio/test/{f}_{idx}.wav")
                input_values = processor(audio, sampling_rate = sr, return_tensors="pt").input_values
                with torch.no_grad():
                    logits = model(input_values.to(device)).logits
                predicted_ids = torch.argmax(logits, dim=-1)

                text = processor.decode(predicted_ids[0])
                with open(os.path.join(manifest_dir, f"{split2kaldi[split]}.wrd"), "a") as f:
                    print(text, file=f)
                with open(os.path.join(manifest_dir, f"{split2kaldi[split]}.ltr"), "a") as f:
                    print(" ".join(text.replace(" ", "|")), file=f)
                with open(os.path.join(manifest_dir, f"{split2kaldi[split]}.sent"), "a") as f:
                    print(label, file=f)

    for subset in ["fine-tune", "dev", "test"]:
        data = {}
        data["sentence"] = []
        data["label"] = []
        flag = 0
        for line, label in zip(open(os.path.join(manifest_dir, f"{subset}.wrd")).readlines(),open(os.path.join(manifest_dir, f"{subset}.sent")).readlines()):
            if(not len(line.strip())): continue
            data["sentence"].append(line.strip())
            data["label"].append(label.strip())

        df = pd.DataFrame(data=data)
        df = df.dropna()
        output_filename = os.path.join(manifest_dir, f"{subset}.huggingface.csv")
        try:
            df.to_csv(output_filename, index=False)
            print(f"Successfully generated file at {output_filename}")

        except:
            print(f"something wrong when generating {output_filename}")
            return


if __name__ == "__main__":
    fire.Fire()
