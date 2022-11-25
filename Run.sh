#!/bin/bash
#$ -N de-cnn
#$ -q gpu2
#$ -m beas
module load cuda/9.0
source activate p36
cd rango/de
python main.py

