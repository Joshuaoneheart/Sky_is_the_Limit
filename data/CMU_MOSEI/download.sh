#!/bin/bash
gdown --id 1gzRgGRB7HOTfnoqNaRXfQHBC_HOamCPD
unzip CMU_MOSEI.zip
unzip CMU_MOSEI/Transcript.zip
unzip CMU_MOSEI/Segmented_Audio.zip
rm -rf CMU_MOSEI.zip
rm -rf CMU_MOSEI/Transcript.zip
rm -rf CMU_MOSEI/Segmented_Audio.zip
mv CMU_MOSEI/* .
mv Segmented_Audio/ Audio
mv Segmented/ Transcript
rmdir CMU_MOSEI
