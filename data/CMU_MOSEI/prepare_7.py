import os
import fire
import numpy as np
import pandas as pd
import soundfile

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

splits = {"train", "dev", "test"}

def create_manifest(
    data_dir="",
    manifest_dir="manifest_7"
):
    os.makedirs(manifest_dir, exist_ok=True)

    labels = sorted([3,2,1,0,-1,-2,-3])
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
                df["file"].array, df["index"].array, df["start"].array, df["end"].array, df["label7"].array,df["split"].array
            ):
                if(split2idx[split] != sp): continue
                frames = int(16000 * (end - start))
                print(f"{f}_{idx}.wav\t{frames}", file=fp)
                text = ""
                text_path = os.path.join(data_dir, f"Transcript/Combined/{f}.txt")
                with open(text_path, "r") as in_f:
                  for line in in_f.readlines():
                    if(line.split("___")[1] == str(idx)):
                      text = line.split("___")[-1].strip()
                      break
                text = text.replace("\"", "").replace("'", "")
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
        for line, label in zip(open(os.path.join(manifest_dir, f"{subset}.wrd")).readlines(),open(os.path.join(manifest_dir, f"{subset}.sent")).readlines()):
            if(not len(line.strip())): continue
            data["sentence"].append(line.strip())
            data["label"].append(label.strip())

        df = pd.DataFrame(data=data)
        output_filename = os.path.join(manifest_dir, f"{subset}.huggingface.csv")
        try:
            df.to_csv(output_filename, index=False)
            print(f"Successfully generated file at {output_filename}")

        except:
            print(f"something wrong when generating {output_filename}")
            return

if __name__ == "__main__":
    fire.Fire()
