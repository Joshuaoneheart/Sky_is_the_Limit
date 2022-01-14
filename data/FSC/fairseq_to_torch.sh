#!/bin/bash
python3 -m transformers.models.wav2vec2.convert_wav2vec2_original_pytorch_checkpoint_to_pytorch --checkpoint_path ./save/checkpoints/checkpoint_best.pt --dict_path ./manifest/dict.ltr.txt --pytorch_dump_folder_path "./torch_ckpt"
