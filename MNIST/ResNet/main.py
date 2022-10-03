import tensorflow as tf
import os, argparse
import numpy as np
from Model import ResNetModel
from DataLoader import load_data, train_valid_split
from Configure import model_configs, training_configs


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("--save_dir", help="path to save the results")
parser.add_argument("result_dir", help="path to the result")
args = parser.parse_args()

if __name__ == '__main__':
	sess = tf.Session()
	model = ResNetModel(sess, model_configs)

	if args.mode == 'train':
		x_train, y_train, x_test, y_test = load_data()
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
		model.train(x_train[0:200, :], y_train[0:200], training_configs, x_valid[0:10, :], y_valid[0:10])
	else:
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data()
		model.evaluate(x_test, y_test, [30])



