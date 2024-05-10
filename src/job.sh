source /data/arguellesa/venv/traice/bin/activate
module load python39
if idlegpu 2>/dev/null 1>&2 ; then
        export CUDA_VISIBLE_DEVICES=idlegpu -p 1;
fi;
# to train
python model/train.py
# to test
# python model/test.py