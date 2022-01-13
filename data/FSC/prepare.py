import os
import fire
import numpy as np
import pandas as pd
import soundfile

from multiprocessing import Pool
from tqdm.auto import tqdm

split2kaldi = {
    "train": "fine-tune",
    "valid": "dev",
    "test": "test",
}

splits = {"train", "valid", "test"}

def create_manifest(
    data_dir="",
    manifest_dir="manifest"
):
    os.makedirs(manifest_dir, exist_ok=True)

    actions = ['change language', 'activate', 'deactivate', 'increase', 'decrease', 'bring']
    objects = ['none', 'music', 'lights', 'volume', 'heat', 'lamp', 'newspaper', 'juice', 'socks', 'shoes', 'Chinese', 'Korean', 'English', 'German']
    locations = ['none', 'kitchen', 'bedroom', 'washroom']
    labels = []
    for action in actions:
        for object in objects:
            for location in locations:
                labels.append(f"{action}|{object}|{location}")
    labels.sort()
    with open(os.path.join(manifest_dir, f"labels.sent.txt"), "w") as f:
        for l in labels:
            print(l, file=f)
    for split in splits:
        metadata_path = os.path.join(data_dir, f"FSC/data/{split}_data.csv")
        df = pd.read_csv(metadata_path)
        print(df.iloc[0])

        with open(os.path.join(manifest_dir, f"{split2kaldi[split]}.tsv"), "w") as fp:
            print(os.path.abspath(os.path.join(data_dir, "FSC/wavs/")), file=fp)
            for p, t, a, o, l in zip(
                df["path"].array, df["transcription"].array,  df["action"].array, df["object"].array, df["location"].array
            ):
                p = p.split("/")[-1]
                s, _ = soundfile.read(os.path.join(data_dir, "FSC/wavs", p))
                frames = len(s)
                print(f"{p}\t{frames}", file=fp)
                text = ""
                with open(os.path.join(manifest_dir, f"{split2kaldi[split]}.wrd"), "a") as f:
                    print(t, file=f)
                with open(os.path.join(manifest_dir, f"{split2kaldi[split]}.ltr"), "a") as f:
                    print(" ".join(t.replace(" ", "|")), file=f)
                with open(os.path.join(manifest_dir, f"{split2kaldi[split]}.sent"), "a") as f:
                    print(f"{a}|{o}|{l}", file=f)

    for subset in ["fine-tune", "dev", "test"]:
        data = {}
        data["sentence"] = []
        data["label"] = []
        for line in open(os.path.join(manifest_dir, f"{subset}.wrd")).readlines():
            data["sentence"].append(line.strip())
        for line in open(os.path.join(manifest_dir, f"{subset}.sent")).readlines():
            data["label"].append(line.strip())

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
