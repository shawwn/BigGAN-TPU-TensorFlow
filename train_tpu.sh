#!/bin/bash

#nohup pipenv run python3.6 main_tpu.py --use-tpu \
exec python3 main_tpu.py --use-tpu \
	--train-input-path gs://gpt-2-poetry/data/imagenet/out/train* \
	--eval-input-path gs://gpt-2-poetry/data/imagenet/out/validation* \
	--model-dir gs://gpt-2-poetry/gan/imagenet/model \
	--result-dir ./results \
	--batch-size 256  \
	--ch 64 \
	--self-attn-res 64 \
	--g-lr 0.0001 \
	--d-lr 0.0004 \
	--verbosity INFO \
	--train-examples 1281167 \
	--eval-examples 50000 \
	--tag sagan \
	--tag run-$RANDOM \
	$@

