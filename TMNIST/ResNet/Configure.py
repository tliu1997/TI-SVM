"""This script setting configurations
"""

model_configs = {
	"name": 'ResNetModel',
	"resnet_version": 2,
	"num_classes": 10,
	"first_num_filters": 16,
	"resnet_size": 18,
	'modeldir': 'result',
	"weight_decay": 2e-4
}

training_configs = {
	"batch_size": 64,
	"save_interval": 30,
	"max_epoch": 30
}