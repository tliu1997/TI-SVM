## **Learning from Small Samples: Transformation-Invariant SVMs with Composition and Locality at Multiple Scales**
[*Code for ''Learning from Small Samples: Transformation-Invariant SVMs with Composition and Locality at Multiple Scales'' (NeurIPS 2022)*](https://arxiv.org/abs/2109.12784)


### Prerequisites
Here we list our running environment:
- python == 3.7.6
- numpy == 1.19.0
- scipy == 1.2.1
- tensorflow == 1.14.0
- keras == 2.3.1
- scikit-learn == 0.22.1
- emnist == 0.0
- matplotlib == 3.0.3
- tqdm == 4.62.2


### Dataset
The ratio of training set and validation set is always 5:1. 
In addition, the number of test samples is always the entire test set, which is 10,000 for the MNIST dataset and 20,800 for the EMNIST Letter dataset.
The randomness of splits is fixed (random_state=4), while translated pixels and rotated degrees (transformed datasets) are random (need to run 10 times to calculate the average and standard deviation).


### Training and testing
To run the experiments, simply execute the following commands, 
```
python main.py
```

