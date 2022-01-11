#!/bin/bash
cd data/CMU_MOSEI/
./download.sh
cd ../ASRGLUE/ 
./download.sh
cd ../FSC/
./download.sh
git clone https://github.com/pytorch/fairseq.git
cd fairseq && pip install --editable ./
git clone https://github.com/asappresearch/slue-toolkit.git
