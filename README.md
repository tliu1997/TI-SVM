## **Learning from Small Samples: Transformation-Invariant SVMs with Composition and Locality at Multiple Scales**
[*Code for ''Learning from Small Samples: Transformation-Invariant SVMs with Composition and Locality at Multiple Scales'' (NeurIPS 2022)*](https://arxiv.org/abs/2109.12784)


### Prerequisites
Here we list our running environment:
- python == 3.7.13
- PyTorch == 1.12.1
- pytorch-lightning == 1.7.7
- torchvision == 0.13.1
- numpy == 1.21.6
- scipy == 1.7.3
- scikit-learn == 1.0.2
- emnist == 0.0
- matplotlib == 3.5.3
- tqdm == 4.64.1


### Dataset
The ratio of training set and validation set is always 5:1. 
In addition, the number of test samples is always the entire test set, which is 10,000 for the MNIST/Transformed MNIST dataset and 20,800 for the EMNIST Letter dataset.
The randomness of splits is fixed (random_state=4), while translated pixels and rotated degrees (transformed datasets) are random (need to run 5 times to calculate the average and standard deviation).


### Training and testing
To run the experiments, simply execute the following commands, 
```
python main.py
```

