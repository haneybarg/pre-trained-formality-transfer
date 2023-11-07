#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50000

python train.py -style 0 -ratio 1.0 -dataset $1 -order $2.0 -$3 -$4
python infer.py -style 0 -dataset $1 -order $2.0
# rm checkpoints/bart_$1_$2.0_0.chkpt

echo "----------------Style----------------"
python classifier/test.py -dataset $1 -order $2.0

echo "----------------BLEU----------------"
python utils/tokenizer.py data/$1/test/formal.ref0 data/$1/original_ref/formal.ref0 False
python utils/tokenizer.py data/$1/test/formal.ref1 data/$1/original_ref/formal.ref1 False
python utils/tokenizer.py data/$1/test/formal.ref2 data/$1/original_ref/formal.ref2 False
python utils/tokenizer.py data/$1/test/formal.ref3 data/$1/original_ref/formal.ref3 False

python utils/tokenizer.py outputs/bart_$1.0.txt outputs/bart_$1_$2.0.0 False
perl utils/multi-bleu.perl data/$1/original_ref/$5.ref < outputs/bart_$1_$2.0.0

# echo "----------------BLEURT----------------"
python utils/cal_bleurt.py outputs/bart_$1_$2.0.0 outputs/bart_$1_$2.0.1 \
                         data/$1/test/$5.ref data/$1/test/$6.ref

