
from progress.bar import Bar
import time
import pickle
import traceback


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

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
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, batch_size, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        i = 0
        while i < epochs:
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
                        time = time.time()
                        output = layer.forward_propagation(output)
                        time2 = time.time()
                        print("Finished Layer in ", time2-time)

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
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    def showcase(self):
        for layer in self.layers:
            print(layer.input_shape, layer.output_shape)

    def save(self, name):
        pickle.dump( self.layers, open( name+".p", "wb" ) )

    def load(self, name):
        self.layers = pickle.load( open( name+".p", "rb" ) )
