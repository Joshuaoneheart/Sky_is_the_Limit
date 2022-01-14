#!/bin/bash

#install transfomers
if ! [ -d transformers ]; then
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install -e .
    cd -
fi

#fine-tuning
nlp_modelname=deberta-large
python3 transformers/examples/pytorch/text-classification/run_glue_no_trainer.py \
    --train_file manifest_2/fine-tune.huggingface.csv \
    --validation_file manifest_2/dev.huggingface.csv \
    --model_name_or_path microsoft/${nlp_modelname} \
    --output_dir save/sentiment/nlp_topline_${nlp_modelname}_2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 8 \
    --seed 7
