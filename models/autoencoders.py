import numpy as np
import cPickle
import theano
import theano.tensor as T
import time
import os
from munging.loaders import load_configuration, UnsupervisedLoader


class Autoencoder(object):
    def __init__(self, input_size, hidden_size, hidden_activation=T.nnet.sigmoid, output_activation=lambda x: x,
                 learning_rate=1.e-3, momentum_factor=0.9, iterations=100000, log_step=100, decay=0., params_path=None):
        """
        Sparse autoencoder implementation - base class for all other shallow autoencoders.
        """
        self.params_path = params_path
        self.iterations = iterations
        self.momentum_factor = momentum_factor
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.log_step = log_step
        self.decay = decay

        self.ts_weights = theano.shared(np.random.normal(size=(input_size, hidden_size)).astype(theano.config.floatX),
                                        name='weights', borrow=True)
        self.ts_hidden_bias = theano.shared(np.random.normal(size=(hidden_size, )).astype(theano.config.floatX),
                                            name='hidden_bias', borrow=True)
        self.ts_visible_bias = theano.shared(np.random.normal(size=(input_size, )).astype(theano.config.floatX),
                                             name='visible_bias', borrow=True)

        # momentum speeds
        ts_v_w = theano.shared(np.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='v_w', borrow=True)
        ts_v_hb = theano.shared(np.zeros((hidden_size, ), dtype=theano.config.floatX), name='v_hb', borrow=True)
        ts_v_vb = theano.shared(np.zeros((input_size, ), dtype=theano.config.floatX), name='v_vb', borrow=True)

        t_x = T.matrix(name='data')
        t_y = T.matrix(name='ground_truth')
        t_lr = T.scalar(name='lr')
        t_mf = T.scalar(name='mf')

        cost = self.cost_function(t_x, t_y)
        params = self.ts_weights, self.ts_hidden_bias, self.ts_visible_bias
        gradient = [T.grad(cost, param) for param in params]

        speeds = ts_v_w, ts_v_hb, ts_v_vb

        t_v_updates = [t_mf * v - t_lr * grad for v, grad in zip(speeds, gradient)]

        updates = [(v, update) for v, update in zip(speeds, t_v_updates)]

        updates.extend([(param, param + update) for param, update in zip(params, t_v_updates)])

        print 'Compiling theano functions'

        self.tf_train = theano.function(inputs=[t_x, t_y, t_lr, t_mf], updates=updates)

        self.tf_reconstruction_error = theano.function(inputs=[t_x, t_y], outputs=cost)

        self.tf_hidden = theano.function(inputs=[t_x], outputs=self.hidden(t_x))
        self.tf_output = theano.function(inputs=[t_x], outputs=self.output(t_x))

    def hidden(self, x):
        return self.hidden_activation(T.dot(x, self.ts_weights) + self.ts_hidden_bias)

    def output(self, hidden):
        return self.output_activation(T.dot(hidden, self.ts_weights.T) + self.ts_visible_bias)

    def cost_function(self, x, y):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return T.mean((y - output) ** 2)

    def train(self, generator):
        print 'Start training...'
        start = time.time()
        for (mini_batch, gt_batch), iteration in zip(generator, xrange(self.iterations)):
            self.tf_train(mini_batch, gt_batch, self.learning_rate, self.momentum_factor)
            self.learning_rate *= (1. - self.decay) / 1.
            if (iteration + 1) % self.log_step == 0:
                end = time.time()
                cost = self.tf_reconstruction_error(mini_batch, gt_batch)
                print "Iteration %d - reconstruction error: %f. Took %f seconds" % (iteration + 1, cost, end - start)
                start = time.time()
        return self

    @property
    def params(self):
        return [param.get_value() for param in [self.ts_weights, self.ts_hidden_bias, self.ts_visible_bias]]

    @params.setter
    def params(self, value):
        for param, val in zip([self.ts_weights, self.ts_hidden_bias, self.ts_visible_bias], value):
            param.set_value(val)


class ContractiveAutoencoder(Autoencoder):
    def __init__(self, input_size, hidden_size, **kwargs):
        self.lambda_ = kwargs['lambda_']
        kwargs.pop('lambda_', None)
        super(ContractiveAutoencoder, self).__init__(input_size, hidden_size, **kwargs)

    def cost_function(self, x, y):
        cost = super(ContractiveAutoencoder, self).cost_function(x, y)
        jacobian = T.sum(T.grad(cost, x) ** 2)
        return self.lambda_ * jacobian + cost


class StackedAutoencoders(object):
    def __init__(self, config, mini_batch_size=128, warm_start=False, start_from=0):
        self.mini_batch_size = mini_batch_size
        self.caes = []
        for _, params in sorted(config.items()):
            cae = ContractiveAutoencoder(input_size=params['input_size'], hidden_size=params['hidden_size'],
                                         learning_rate=params['lr'], momentum_factor=params['mf'],
                                         lambda_=params['lambda'], decay=params['decay'],
                                         iterations=params['iterations'], params_path=params['params_path'])
            if warm_start:
                if os.path.exists(params['params_path']):
                    with open(params['params_path'], 'rb') as f:
                        cae.params = cPickle.load(f)
            self.caes.append(cae)
        self.start_from = start_from

    def train(self, generator):
        for i, cae in enumerate(self.caes):
            if not (i == self.start_from):
                if i > 0:
                    preprocessor = self.caes[i - 1]
                    generator.update_data(preprocessor)
                continue
            print "Training %d/%d autoencoders" % (i + 1, len(self.caes))
            preprocessor = None if i == 0 else self.caes[i - 1]
            cae = cae.train(
                generator.generate(mini_batch_size=self.mini_batch_size, preprocessor=preprocessor))
            print "Dumping parameters..."
            with open(cae.params_path, 'w') as f:
                cPickle.dump(cae.params, f)


def train_stacked_aes():
    mini_batch_size = 128
    generator = UnsupervisedLoader('../data')
    config = load_configuration('../config/caes.json')
    scaes = StackedAutoencoders(config, mini_batch_size=mini_batch_size, warm_start=True, start_from=2)

    scaes.train(generator)


if __name__ == "__main__":
    train_stacked_aes()