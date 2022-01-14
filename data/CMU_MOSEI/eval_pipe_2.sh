#!/bin/bash

python3 ../../eval/eval_nlp_sentiment.py \
        --data manifest_asr_2/ \
        --subset test \
        --save-dir save/nlp_topline_deberta-large_2 \
        --use-gpu \
        --eval
