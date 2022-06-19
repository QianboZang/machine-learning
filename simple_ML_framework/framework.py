#!/usr/bin/env python3

## You will have to implement a very primitive machine learning framework. The skeleton is given. You have to fill the
## blocks marked by "## Implement". The goal is to provide you an insight into how machine learning frameworks work
## while practicing the most important details from the class.
##
## You have to implement the following modules (or call them layers?):
##      * Activation functions: Tanh, Softmax
##      * Layers: Linear, Sequential
##      * Losses: Cross-Entropy
##
## The linear layer is also called perceptron, fully connected layer, etc. The bias term is not included in the
## network weight matrix W because of performance reasons (concatenating the 1 to the input are slow). This is the case
## for the real-world ML frameworks too.
##
## All the functions you have to implement has their signature pre-specified, with annotated types. You should _not_
## change this, because it will be tested by an automated system which depends on the current interface.
##
## The sequential layer receives a list of layers for the constructor argument, and calls them in order on forward
## and in reverse order on backward. It is just syntactic sugar that makes the code much nicer. For example, you can
## create a two-layer neural network with tanh activation functions with net = Sequential([Linear(5, 8), Tanh(),
## Linear(8, 3), Tanh()]) and then run a forward pass using output = net.forward(your data).
##
## All the modules have a forward() and a backward() function. forward() receives one argument (except for the loss) and
## returns that layer's output. The backward() receives the dL/dout, flowing back on the output of the layer, and
## should return a BackwardResult object with the following fields:
##      variable_grads: a dict of gradients, where the keys are the same as in the keys in the layer's .var. The
##                      values are numpy arrays representing the gradient for that variable.
##      input_grads: a numpy array representing the gradient for the input (that was passed to the layer in the forward
##                   pass).
##
## The backward() does not receive the forward pass's input, although it might be needed for the gradient
## calculation. You should save them in the forward pass for later use in the backward pass. You don't have to worry
## about most of this, as it is already implemented in the skeleton. There are 2 important takeaways: you have to
## calculate the gradient of both of your variables and the layer input in the backward pass, and if you need to reuse
## the input from the forward pass, you need to save it.
##
## You will also have to implement the function train_one_step(), which does one step of weight update based on the
## training data and the learning rate. It should run backpropagation, followed by gradient descent.
##
## Optionally you can implement gradient checking, enabling you to be almost sure that your backward functions are
## correct. To do this, you will have to fill in the analytic and numerical gradient computation part of the
## gradient_check() function. This does iterates over all the elements of all variables, nudges it a bit in both
## directions, and recalculates the network output. Based on that, you can calculate what the gradient should be if we
## assume that the forward pass is correct. The method is known as numerical differentiation, specifically the
## symmetric difference quotient. If your gradient checking passes and your error is around 0.008, your solution is
## probably correct. It's worth trying to intentionally corrupt the backward of some module (for example, Tanh) by a
## tiny bit (~0.01) and see if the grad check fails. If not, your grad check might be wrong.
##
## Finally, you would have to complete the create_network() function, which should return a Sequential neural network of
## 3 layers: a tanh input layer with 2 inputs and 50 outputs, a tanh hidden layer with 30 outputs, and finally a softmax
## output layer with 2 outputs (usually for two-way classification we don't use softmax, but we will need it for the
## MNIST part anyway, so we use it here as well).
##
## At the end of the training, your loss should be around 0.05. Don't be afraid if it differs a bit, but a
## significantly higher value may indicate a problem.
##
## There are asserts at many points in the code there that will check the shapes of the gradients. Remember: the
## gradient for a variable must have the same shape as the variable itself. Imagine the variables and the network
## inputs/outputs as a cable with a given number of wires: no matter in which direction you pass the data, the number
## of wires is the same.
##
## Please do your calculations in a vectorized way. Otherwise, it will be painfully slow. You have to use for loop only
## twice in this file.
##
## Please install the dependencies needed for this script with pip3 -r requirements.txt
##
## Good luck, I hope you'll enjoy it :)!


import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


# Interface definitions
class Layer:
    var: Dict[str, np.ndarray] = {}

    @dataclass
    class BackwardResult:
        variable_grads: Dict[str, np.ndarray]
        input_grads: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, error: np.ndarray) -> BackwardResult:
        raise NotImplementedError()


class Loss:
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError()

    def backward(self) -> np.ndarray:
        raise NotImplementedError()


# Implementation starts
class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        ## Implement

        # result =
        result = np.tanh(x)

        ## End
        self.saved_variables = {
            "result": result
        }
        return result

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        tanh_x = self.saved_variables["result"]

        ## Implement

        # d_x =
        d_x = (1-tanh_x**2)*grad_in

        ## End
        assert d_x.shape == tanh_x.shape, "Input: grad shape differs: %s %s" % (d_x.shape, tanh_x.shape)

        self.saved_variables = None
        return Layer.BackwardResult({}, d_x)


class Softmax(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        ## Implement

        # result =
        x = x - np.max(x)
        exp_x = np.exp(x)
        result = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        ## End
        self.saved_variables = {
            "result": result
        }
        return result

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        softmax = self.saved_variables["result"]

        ## Implement

        # d_x =
        d_x = softmax*(grad_in-(grad_in*softmax).sum(axis=1)[:, None])

        ## End
        assert d_x.shape == softmax.shape, "Input: grad shape differs: %s %s" % (d_x.shape, softmax.shape)

        self.saved_variables = None
        return Layer.BackwardResult({}, d_x)


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.var = {
            "W": np.random.normal(0, np.sqrt(2 / (input_size + output_size)), (input_size, output_size)),
            "b": np.zeros((output_size), dtype=np.float32)
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        W = self.var['W']
        b = self.var['b']

        ## Implement
        ## Save your variables needed in backward pass to self.saved_variables.

        # y =
        y = np.matmul(x, W) + b

        self.saved_variables = {
            "input": x,
            "output": y,
            "W": W,
            "b": b
        }

        ## End
        return y

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        ## Implement

        # dW =
        # db =
        W = self.saved_variables['W']
        x = self.saved_variables['input']
        # d_inputs =
        dW = np.matmul(x.T, grad_in)
        d_temp = np.ones(x.shape[0])
        db = np.matmul(d_temp, grad_in)
        d_inputs = np.matmul(grad_in, W.T)

        ## End
        assert d_inputs.shape == x.shape, "Input: grad shape differs: %s %s" % (d_inputs.shape, x.shape)
        assert dW.shape == self.var["W"].shape, "W: grad shape differs: %s %s" % (dW.shape, self.var["W"].shape)
        assert db.shape == self.var["b"].shape, "b: grad shape differs: %s %s" % (db.shape, self.var["b"].shape)

        self.saved_variables = None
        updates = {"W": dW,
                   "b": db}
        return Layer.BackwardResult(updates, d_inputs)


class Sequential(Layer):
    class RefDict(dict):
        def __setitem__(self, k, v):
            assert k in self, "Trying to set a non-existing variable %s" % k
            ref = super().__getitem__(k)
            ref[0][ref[1]] = v

        def __getitem__(self, k):
            ref = super().__getitem__(k)
            return ref[0][ref[1]]

        def items(self) -> Tuple[str, np.ndarray]:
            for k in self.keys():
                yield k, self[k]

    def __init__(self, list_of_modules: List[Layer]):
        self.modules = list_of_modules

        refs = {}
        for i, m in enumerate(self.modules):
            refs.update({"mod_%d.%s" % (i,k): (m.var, k) for k in m.var.keys()})

        self.var = self.RefDict(refs)

    def forward(self, input: np.ndarray) -> np.ndarray:
        ## Implement
        x = input
        l = len(self.modules)
        for module_index in range(l):
            module = self.modules[module_index]
            x = module.forward(x)

        ## End
        return x

    def backward(self, grad_in: np.ndarray) -> Layer.BackwardResult:
        variable_grads = {}

        for module_index in reversed(range(len(self.modules))):
            module = self.modules[module_index]

            ## Implement

            # grads =
            grads = module.backward(grad_in)

            ## End
            grad_in = grads.input_grads
            variable_grads.update({"mod_%d.%s" % (module_index, k): v for k, v in grads.variable_grads.items()})

        return Layer.BackwardResult(variable_grads, grad_in)


class CrossEntropy(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        Y = prediction
        T = target
        n = prediction.size

        ## Implement
        ## The loss function has to return a single scalar, so we have to take the mean over the batch dimension.
        ## Don't forget to save your variables needed for backward to self.saved_variables.
        # mean_ce =
        mean_ce = -np.sum(T*np.log(Y))/n
        self.saved_variables = {
            "y": Y,
            "t": T,
            "mean_ce": mean_ce,
            "n": n
        }

        ## End
        return mean_ce

    def backward(self) -> np.ndarray:
        ## Implement
        y = self.saved_variables["y"]
        t = self.saved_variables["t"]
        n = self.saved_variables["n"]

        # d_prediction =
        d_prediction = -t/(y*n)

        ## End
        assert d_prediction.shape == y.shape, "Error shape doesn't match prediction: %d %d" % \
                                              (d_prediction.shape, y.shape)

        self.saved_variables = None
        return d_prediction


def train_one_step(model: Layer, loss: Loss, learning_rate: float, input: np.ndarray, target: np.ndarray) -> float:
    ## Implement
    prediction = model.forward(input)
    loss_value = loss.forward(prediction, target)
    grads = model.backward(loss.backward()).variable_grads

    for key, value in grads.items():
        model.var[key] = model.var[key]-learning_rate*value

    ## End
    return loss_value


def create_network() -> Layer:
    ## Implement
    network = Sequential([Linear(2, 50), Tanh(), Linear(50, 30), Tanh(),
                          Linear(30, 2), Softmax()])

    ## End
    return network


def gradient_check():
    X, T = twospirals(n_points=10)
    NN = create_network()
    eps = 0.0001

    loss = CrossEntropy()
    loss.forward(NN.forward(X), T)
    variable_gradients = NN.backward(loss.backward()).variable_grads

    all_succeeded = True

    # Check all variables. Variables will be flattened (reshape(-1)), in order to be able to generate a single index.
    for key, value in NN.var.items():
        variable = NN.var[key].reshape(-1)
        variable_gradient = variable_gradients[key].reshape(-1)
        success = True

        if NN.var[key].shape != variable_gradients[key].shape:
            print("[FAIL]: %s: Shape differs: %s %s" % (key, NN.var[key].shape, variable_gradients[key].shape))
            success = False
            break

        # Check all elements in the variable
        for index in range(variable.shape[0]):
            var_backup = variable[index]

            ## Implement

            # analytic_grad =
            # numeric_grad =
            analytic_grad = variable_gradient[index]
            variable[index] = var_backup-eps
            error_left = loss.forward(NN.forward(X), T)
            variable[index] = var_backup+eps
            error_right = loss.forward(NN.forward(X), T)
            numeric_grad = (error_right-error_left)/(2*eps)

            ## End

            variable[index] = var_backup
            if abs(numeric_grad - analytic_grad) > 0.00001:
                print("[FAIL]: %s: Grad differs: numerical: %f, analytical %f" % (key, numeric_grad, analytic_grad))
                success = False
                break

        if success:
            print("[OK]: %s" % key)

        all_succeeded = all_succeeded and success

    return all_succeeded


###############################################################################################################
# Nothing to do past this line.
###############################################################################################################

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    np.random.seed(12345)

    plt.ion()


    def twospirals(n_points=120, noise=1.6, twist=420):
        """
         Returns a two spirals dataset.
        """
        np.random.seed(0)
        n = np.sqrt(np.random.rand(n_points, 1)) * twist * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
        d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
        X, T = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
                np.hstack((np.zeros(n_points), np.ones(n_points))))
        T = np.reshape(T, (T.shape[0], 1))
        T = np.concatenate([T, 1-T], axis=1)
        return X, T


    fig, ax = plt.subplots()


    def plot_data(X, T):
        ax.scatter(X[:, 0], X[:, 1], s=40, c=T[:, 0], cmap=plt.cm.Spectral)


    def plot_boundary(model, X, targets, threshold=0.0):
        ax.clear()
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        y = model.forward(X_grid)[:, 0]
        ax.contourf(xx, yy, y.reshape(*xx.shape) < threshold, alpha=0.5)
        plot_data(X, targets)
        ax.set_ylim([y_min, y_max])
        ax.set_xlim([x_min, x_max])
        plt.show()
        plt.draw()
        plt.pause(0.001)


    def main():
        print("Checking the network")
        if not gradient_check():
            print("Failed. Not training, because your gradients are not good.")
            return
        print("Done. Training...")

        X, T = twospirals(n_points=200, noise=1.6, twist=600)
        NN = create_network()
        loss = CrossEntropy()

        learning_rate = 0.02

        for i in range(20000):
            curr_error = train_one_step(NN, loss, learning_rate, X, T)
            if i % 200 == 0:
                print("step: ", i, " cost: ", curr_error)
                plot_boundary(NN, X, T, 0.5)

        plot_boundary(NN, X, T, 0.5)
        print("Done. Close window to quit.")
        plt.ioff()
        plt.show()



    main()