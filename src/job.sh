#!/bin/bash
<<<<<<< HEAD
source traicenv/bin/activate
module load python39
# to train
# python main.py train model/.config
# to test
python main.py model/.config /data/arguellesa/traice/old_weights/checkpoint-500 /data/arguellesa/traice/1.txt
=======
source ../traicenv/bin/activate
module load python39
# to train uncomment next line
# python model/train.py
# to test uncomment next line
python model/test.py /data/arguellesa/traice/old_weights/checkpoint-500 /data/arguellesa/traice/1.txt 
>>>>>>> 303ecb4c2448c01f6cde61f644edca3c8208a24c
