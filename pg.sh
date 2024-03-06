#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50000

# python train.py -style 0 -ratio 1.0 -dataset $1 -order $2.0 -$3 -$4
# python infer.py -style 0 -dataset $1 -order $2.0
# rm checkpoints/bart_$1_$2.0_0.chkpt

echo "----------------Style----------------"
# check which dataset they are using
python classifier/test.py -dataset $1 -order $2.0 -data $7

echo "----------------BLEU----------------"
python utils/tokenizer.py $8 data/$1/original_ref/formal.ref False
python utils/tokenizer.py $7 outputs/bart_$1_$2.0.0 False
perl utils/multi-bleu.perl data/$1/original_ref/$5.ref < outputs/bart_$1_$2.0.0

# echo "----------------BLEURT----------------"

python utils/cal_bleurt.py $7 outputs/bart_$1_$2.0.1 \
                        data/$1/original_ref/formal.ref data/$1/test/$6.ref

