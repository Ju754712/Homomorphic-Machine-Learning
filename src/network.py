
from progress.bar import Bar
import time
import pickle
import traceback
import random

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)
    def remove(self, layer):
        self.layers.pop(layer)
    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            # bar = Bar('Processing plain data', max=len(self.layers))
            for layer in self.layers:
                # bar.next()
                output = layer.forward_propagation(output)
            result.append(output)
            # bar.finish()
        return result

    def predict_more(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            # bar = Bar('Processing encrypted data', max=len(self.layers))
            for layer in self.layers:
                # bar.next()        
                output = layer.forward_propagation_more(output)
            result.append(output)

            # bar.finish()
        return result

    def predict_ckks(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            # bar = Bar('Processing encrypted data', max=len(self.layers))
            for layer in self.layers:
                # bar.next()        
                output = layer.forward_propagation_ckks(output)
            result.append(output)

            # bar.finish()
        return result

    # train the network
    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, shuffle = False, adaptive=False):
        # sample dimension first
        samples = len(x_train)

        # training loop
        i = 1
        while i <= epochs:
            if shuffle:
                idxs = [i for i in range(len(x_train))]
                random.shuffle(idxs)
                x_train = x_train[idxs]
                y_train = y_train[idxs]
            if adaptive:
                learning_rate = learning_rate*(1-adaptive)
                print(learning_rate)
            print("Epoch ",i)
            err = 0
            j = 0
            batches = int(samples/batch_size)
            bar = Bar('Processing Batch', max=batches)
            while j < batches:
                k = 0
                error = 0
                while k < batch_size:
                    output = x_train[j*batch_size+k]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                        # compute loss (for display purpose only)
                    err += self.loss(y_train[j*batch_size+k], output)
                    # Compute Gradient
                    error += self.loss_prime(y_train[j*batch_size+k], output)
                    k+=1
                # Average Gradient
                error = error/batch_size
                # backward propagation
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
                bar.next()
                j+=1
            bar.finish()
            i+=1

                

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i-1, epochs, err))

    def fit_more(self, x_train, y_train, epochs, batch_size, learning_rate, shuffle = False, adaptive=False):
        # sample dimension first
        samples = len(x_train)

        # training loop
        i = 1
        while i <= epochs:
            if shuffle:
                idxs = [i for i in range(len(x_train))]
                random.shuffle(idxs)
                x_train = x_train[idxs]
                y_train = y_train[idxs]
            if adaptive:
                learning_rate = learning_rate*(1-adaptive)
                print(learning_rate)
            print("Epoch ",i)
            err = 0
            j = 0
            batches = int(samples/batch_size)
            bar = Bar('Processing Batch', max=batches)
            while j < batches:
                k = 0
                error = 0
                while k < batch_size:
                    output = x_train[j*batch_size+k]
                    for layer in self.layers:
                        output = layer.forward_propagation_more_encrypted(output)
                        # compute loss (for display purpose only)
                    # err += self.loss(y_train[j*batch_size+k], output)
                    # Compute Gradient
                    error += self.loss_prime(y_train[j*batch_size+k], output)
                    k+=1
                # Average Gradient
                error = error * (1/batch_size)
                # backward propagation
                for layer in reversed(self.layers):
                    error = layer.backward_propagation_more(error, learning_rate)
                bar.next()
                j+=1
            bar.finish()
            i+=1

                

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i-1, epochs, err))

    def showcase(self):
        for layer in self.layers:
            print(layer.input_shape, layer.output_shape)

    def save(self, name):
        pickle.dump( self.layers, open( name+".p", "wb" ) )

    def load(self, name):
        self.layers = pickle.load( open( name+".p", "rb" ) )
