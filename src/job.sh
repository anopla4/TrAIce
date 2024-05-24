#!/bin/bash
source ../traicenv/bin/activate
module load python39
# to train uncomment next line
# python model/train.py
# to test uncomment next line
python model/test.py /data/arguellesa/traice/old_weights/checkpoint-500 /data/arguellesa/traice/1.txt 