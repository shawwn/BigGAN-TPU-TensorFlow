
from comet_ml import Experiment

import numpy as np
import os

import math
import os.path
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)
logger.setLevel(1)

from inception_score import prefetch_inception_model
from input import train_input_fn, eval_input_fn, predict_input_fn
from utils import *
from args import *


def run_main_loop(args, train_estimator, predict_estimator, **train_kws):
	total_steps = 0
	train_steps = math.ceil(args.train_examples / args._batch_size)
	eval_steps  = math.ceil(args.eval_examples  / args._batch_size)

	if args.use_comet:
		experiment = Experiment(api_key=comet_ml_api_key, project_name=comet_ml_project, workspace=comet_ml_workspace)
		experiment.log_parameters(vars(args))
		experiment.add_tags(args.tag)
		experiment.set_name(model_name(args))
	else:
		experiment = None

	prefetch_inception_model()

	with tf.gfile.Open(os.path.join(suffixed_folder(args, args.result_dir), "eval.txt"), "a") as eval_file:
		for epoch in range(0, args.epochs, args.predict_every):

			logger.info(f"Training epoch {epoch}")
			train_estimator.train(input_fn=train_input_fn, steps=train_steps * args.predict_every, **train_kws)
			total_steps += train_steps * args.predict_every

			if args.use_comet:
				experiment.set_step(epoch)

			# logger.info(f"Evaluate {epoch}")
			# evaluation = predict_estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
			# logger.info(evaluation)
			# save_evaluation(args, eval_file, evaluation, epoch, total_steps)
			
			# if args.use_comet:
			# 	experiment.log_metrics(evaluation)
			
			logger.info(f"Generate predictions {epoch}")
			predictions = predict_estimator.predict(input_fn=predict_input_fn)
			
			logger.info(f"Save predictions")
			save_predictions(args, suffixed_folder(args, args.result_dir), eval_file, predictions, epoch, total_steps, experiment)

	logger.info(f"Completed {args.epochs} epochs")

