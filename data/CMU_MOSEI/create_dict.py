from collections import Counter
import fire
import os

def main(thres=0):
    output = "manifest/dict.ltr.txt"
    counter = Counter()
    for filename in os.listdir("./Transcript/Combined"):
      with open("./Transcript/Combined/" + filename, "r") as f:
          for line in f:
              counter.update(line.split("___")[-1].strip().split())

    with open(output, "w") as f:
        for tok, count in counter.most_common():
            if count >= thres:
                print(tok, count, file=f)


if __name__ == "__main__":
    fire.Fire(main)
