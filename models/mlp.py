__author__ = 'mateuszopala'
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
import cPickle
from munging.loaders import SupervisedLoader


if __name__ == "__main__":
    with open('../data/params_0.pkl', 'rb') as f:
        w_0, _, _ = cPickle.load(f)

    with open('../data/params_1.pkl', 'rb') as f:
        w_1, _, _ = cPickle.load(f)

    with open('../data/params_2.pkl', 'rb') as f:
        w_2, _, _ = cPickle.load(f)

    train_x, train_y = SupervisedLoader.load('../data')

    model = Sequential()
    model.add(Dense(33, 256, weights=[w_0]))
    model.add(Activation('sigmoid'))
    # model.add(Dropout(0.2))
    model.add(Dense(256, 512, weights=[w_1]))
    model.add(Activation('sigmoid'))
    # model.add(Dropout(0.2))
    model.add(Dense(512, 512, weights=[w_2]))
    model.add(Dropout(0.5))
    model.add(Dense(512, 1, init='normal'))
    model.add(Activation('linear'))

    # sgd = SGD(lr=1.e-5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='adagrad')

    model.fit(train_x, train_y, nb_epoch=500, batch_size=32, validation_split=0.2)