from collections import Counter
import fire
import os

def main(thres=0):
    output_1 = "manifest_2/dict.ltr.txt"
    output_2 = "manifest_7/dict.ltr.txt"
    counter = Counter()
    for filename in os.listdir("./Transcript/Combined"):
      with open("./Transcript/Combined/" + filename, "r") as f:
          for line in f:
              counter.update(line.split("___")[-1].strip().split())

    with open(output_1, "w") as f_1:
      with open(output_2, "w") as f_2:
        for tok, count in counter.most_common():
            if count >= thres:
                print(tok, count, file=f_1)
                print(tok, count, file=f_2)


if __name__ == "__main__":
    fire.Fire(main)
