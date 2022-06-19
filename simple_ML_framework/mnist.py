#!/usr/bin/env python3

## In this assignment, you will learn how to train a model with early stopping with the simple machine learning
## framework's help from the previous assignment. The intent is to show you how to train a machine learning model.
## It is very similar to what you would do in a real ML framework.
##
## We provide some code to download and load MNIST digits. The MNIST comes with a separate test and training set, but
## not a validation set. Your task is to split the official "train" set to train and validation set and train the
## network with early stopping.
##
## Some notes:
##  * an EPOCH is a pass through the whole training set once.
##  * validation accuracy is the percentage of correctly classified digits on the validation set.
##  * early stopping is when you avoid overfitting the model by measuring validation accuracy every N steps (e.g.,
##    every epoch) and stopping your training when your model begins to worsen on the validation set. You can do that by
##    keeping track of the best validation accuracy so far and its epoch, and stopping if the validation
##    accuracy has not improved in the last M steps (e.g., last 10 epochs).
##    (A better way to do this is to keep the weights of the best performing model, but that is harder since you need a
##    way to save and reload weights of the model. We keep it simple instead and use the last, slightly worse model).
##  * test accuracy is the percentage of correctly classified digits on the test set.
##  * watch out: if you load batch_size of data by NumPy slicing, it is not guaranteed that you will actually get
##    batch_size of them: if your array length is not divisable by batch_size, you will get the remainder as the last
##    batch. Take that in account when calculating the percentages: use shape[0] to determine the real number of
##    elements in the current batch of data.
##  * verify() should be used both in the validate() and test() functions for error measurement
##    without code duplication (just the input data should be different).
##  * this is a 10-way classification task, so your network will have 10 outputs, one for each digit. It
##    can be trained by Softmax nonlinearity followed by a Cross-Entropy loss function. So for every image, you get 10
##    outputs. To figure out which one is the correct class, you should find which is the most active.
##  * MNIST gives the labels as integers (e.g. 3, 0, 2, ...). For the Cross-Entropy loss function, you have to convert
##    them into a one-hot format: [0., 0., 0., 1., 0., ...], [1., 0., 0., 0., 0., ...], [0., 0., 1., 0., 0., ...]. The
##    output of the Softmax layer and the targets must have the same size.
##  * MNIST comes with black-and-white binary images, with a background of 0 and foreground of 255. Each image is 28x28
##    matrix. To feed that to the model, we flatten it to a 784 (28*28) length vector and normalize it by 255, so the
##    backround becomes 0 and the foreground 1.0. Labels are integers between 0-9. You don't have to worry about this;
##    it's already done for you. The networks usually like to receive an input in range -1 .. 1 or generally the mean
##    near 0, and the standard deviation near 1 (as the majority of MNIST pixels is black, normalizing it to 0..1 range
##    is good enough).
##  * You have to have the previous task's solution in the same folder as this file, as it will load your previously
##    implemented functions from there.
##
## Your final test accuracy should be close to 95% and should early stop after about 100 epochs.
##
## Scroll to the "# Nothing to do BEFORE this line." line, and let the fun begin! Good luck!


import os
from urllib import request
import gzip
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

import framework as lib


class MNIST:
    FILES = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]

    URL = "http://yann.lecun.com/exdb/mnist/"

    @staticmethod
    def gzload(file, offset):
        with gzip.open(file, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=offset)

    def __init__(self, set, cache="./cache"):
        os.makedirs(cache, exist_ok=True)

        for name in self.FILES:
            path = os.path.join(cache, name)
            if not os.path.isfile(path):
                print("Downloading " + name)
                request.urlretrieve(self.URL + name, path)

        if set=="test":
            f_offset = 2
        elif set=="train":
            f_offset = 0
        else:
            assert False, "Invalid set: "+set

        self.images = self.gzload(os.path.join(cache, self.FILES[f_offset]), 16).reshape(-1,28*28).astype(np.float)/255.0
        self.labels = self.gzload(os.path.join(cache, self.FILES[f_offset+1]), 8)

    def __len__(self) -> int:
        return self.images.shape[0]


train_validation_set = MNIST("train")
test_set = MNIST("test")

n_train = int(0.7 * len(train_validation_set))
print("MNIST:")
print("   Train set size:", n_train)
print("   Validation set size:", len(train_validation_set) - n_train)
print("   Test set size", len(test_set))

np.random.seed(12345)
batch_size = 64

loss = lib.CrossEntropy()
learning_rate = 0.1

model = lib.Sequential([
    lib.Linear(28*28, 20),
    lib.Tanh(),
    lib.Linear(20, 10),
    lib.Softmax()
])

#######################################################################################################################
# Nothing to do BEFORE this line.
#######################################################################################################################

indices = np.random.permutation(len(train_validation_set))

## Implement
## Hint: you should split indices to 2 parts: a training and a validation one. Later when loading a batch of data,
## just iterate over those indices by loading "batch_size" of them at once, and load data from the dataset by
## train_validation_set.images[your_indices[i: i+batch_size]] and
## train_validation_set.labels[your_indices[i: i+batch_size]]

# train_indices =
# validation_indices =
train_indices = indices[:n_train]
validation_indices = indices[n_train:]


## End

def verify(images: np.ndarray, targets: np.ndarray) -> Tuple[int, int]:
    ## Implement
    # decoder
    y = model.forward(images)
    t = np.argmax(y, axis=1)
    num_ok = np.sum(t == targets)
    total_num = len(t)
    ## End
    return num_ok, total_num


def test() -> float:
    accu = 0.0
    count = 0

    for i in range(0, len(test_set), batch_size):
        images = test_set.images[i: i+batch_size]
        labels = test_set.labels[i: i+batch_size]

        ## Implement. Use the verify() function to verify your data.
        num_ok, total_num = verify(images, labels)
        accu += num_ok
        count += total_num
        ## End

    return accu / count * 100.0


def validate() -> float:
    accu = 0.0
    count = 0

    for i in range(0, len(test_set), batch_size):
        v_indices = validation_indices[i: i + batch_size]
        images = train_validation_set.images[v_indices]
        labels = train_validation_set.labels[v_indices]

        ## Implement. Use the verify() function to verify your data.
        num_ok, total_num = verify(images, labels)
        accu += num_ok
        count += total_num
        ## End

    return accu/count * 100.0


## You should update these: best_validation_accuracy is the best validation set accuracy so far, best_epoch is the
## epoch of this best validation accuracy (the later can be initialized by anything, as the accuracy will be for sure
## better than 0, so it will be updated for sure).
best_validation_accuracy = 0
best_epoch = -1

losses = []
validation_accuracies = []

for epoch in range(1000):
    ## Implement
    # loss_value =
    # validation_accuracy =
    net_loss = 0
    l = len(train_indices)
    for i in range(0, l, batch_size):
        t_indices = train_indices[i: i + batch_size]
        images = train_validation_set.images[t_indices]
        labels = train_validation_set.labels[t_indices]
        labels = (np.eye(10)[labels])
        temp = lib.train_one_step(model, loss, learning_rate, images, labels)
        loss_value = temp*len(labels)
        net_loss += loss_value
    net_loss = net_loss/len(train_indices)
    validation_accuracy = validate()
    losses.append(net_loss)
    validation_accuracies.append(validation_accuracy)
    ## End

    print("Epoch %d: loss: %f, validation accuracy: %.2f%%" % (epoch, loss_value, validation_accuracy))

    ## Implement
    ## Hint: you should check if current accuracy is better that the best so far. If it is not, check before how many
    ## iterations ago the best one came, and terminate if it is more than 10. Also update the best_* variables
    ## if needed.
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_epoch = epoch
    elif epoch > best_epoch + 10:
        print('\t early stops', epoch-best_epoch, 'iterations.')
        break
    # end

print("Test set performance: %.2f%%" % test())
fig, ax = plt.subplots()
ax.set_xlabel('epoch')
ax2 = ax.twinx()
ax.plot(losses, color='green', label='losses')
ax2.plot(validation_accuracies, color='red', label='valid accuracies')
ax.set_ylabel('losses', color='green')
ax2.set_ylabel('validation accuracies %', color='red')
fig.legend()
plt.savefig('ml.png')
plt.show()
