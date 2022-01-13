#!/bin/bash
gdown --id 1BHgcu_jHV5yMN5pddFJdsiEmcTIwIdyL
tar -zxvf FSC.tar.gz
mv fluent_speech_commands_dataset FSC
rm -rf FSC.tar.gz
mv ./FSC/wavs/* ./FSC
cd FSC/speakers
mv */* ../wavs
cd ../..
rm -rf ./FSC/speakers
