from collections import Counter
import fire
import os
import pandas as pd

def main(thres=0):
    output = "manifest/dict.ltr.txt"
    counter = Counter()
    for split in ["train", "valid", "test"]:
        for trans in pd.read_csv(f"FSC/data/{split}_data.csv")["transcription"].array:
            counter.update(trans.strip().split())

    with open(output, "w") as f:
        for tok, count in counter.most_common():
            if count >= thres:
                print(tok, count, file=f)


if __name__ == "__main__":
    fire.Fire(main)
