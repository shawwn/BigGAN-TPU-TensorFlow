#!/bin/bash


#run=$RANDOM
#run=${run:-$RANDOM} # 0=5976, 1=20403
run=${run:-$RANDOM} # 0=5976, 1=20403
#batch=32768
#batch=512
batch=2048
images=1281167
#epochs=$(echo 'scale=0; 200000*2048/1281167+1' | bc -l) # 320
epochs=$((200000*2048/$images + 1)) # 320
examples=$(($epochs * $images * 100000)) # 409973440
steps=$(($examples / $batch)) # 12511

set -x
tpu=${tpu:-tpu-v3-512-euw4a-0}

#nohup pipenv run python3.6 main_tpu.py --use-tpu \
exec python3 main_tpu.py --use-tpu --tpu-name $tpu \
	--train-input-path gs://gpt-2-poetry/data/imagenet/out/train* \
	--eval-input-path gs://gpt-2-poetry/data/imagenet/out/validation* \
	--model-dir gs://gpt-2-poetry/gan/imagenet64big/model \
	--checkpoint-dir gs://gpt-2-poetry/gan/imagenet64big/checkpoint-$run \
	--result-dir ./results \
	--batch-size $batch  \
	--ch 96 \
	--layers 5 \
	--img-size 64 \
	--self-attn-res 32 \
	--g-lr 0.00005 \
	--d-lr 0.00020 \
	--verbosity INFO \
	--train-examples $examples \
	--eval-examples 50000 \
	--tag sagan \
	--tag run-$run \
	$@

#nohup pipenv run python3.6 main_tpu.py --use-tpu \
exec python3 main_tpu.py --use-tpu --tpu-name $tpu \
	--train-input-path gs://gpt-2-poetry/data/imagenet/out/train* \
	--eval-input-path gs://gpt-2-poetry/data/imagenet/out/validation* \
	--model-dir gs://gpt-2-poetry/gan/imagenet64big/model \
	--checkpoint-dir gs://gpt-2-poetry/gan/imagenet64big/checkpoint-$run \
	--result-dir ./results \
	--batch-size $batch  \
	--ch 96 \
	--layers 5 \
	--img-size 256 \
	--self-attn-res 64 \
	--g-lr 0.00005 \
	--d-lr 0.00020 \
	--verbosity INFO \
	--train-examples $examples \
	--eval-examples 50000 \
	--tag sagan \
	--tag run-$run \
	$@

exec python3 main_tpu.py \
	--model-dir ./checkpoint-$run/model \
	--checkpoint-dir ./checkpoint-$run \
	--data-source mnist \
	--img-size 28 \
	--img-ch 1 \
	--num-labels 10 \
	--steps-per-loop 500 \
	--train-examples $examples \
	--eval-examples 10000 \
	--batch-size 8 \
	--ch 8 \
	--layers 3 \
	--epoch 1000 \
	--tag mnist \
	--tag run-$run \
	$@
