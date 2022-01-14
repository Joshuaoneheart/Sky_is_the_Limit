#!/bin/bash
python ../../eval_asr.py eval_asr \
--model save/checkpoints/ \
--data manifest \
--subset test
