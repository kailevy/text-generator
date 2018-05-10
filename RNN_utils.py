from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras import losses
import pickle
import tensorflow as tf

fuzz_factor = 1e-7

# method for generating text
def generate_text(model, length, vocab_size, ix_to_char, start, random):
    # starting with random character
    if start == -1:
        ix = [np.random.randint(vocab_size)]
    else:
        ix = start
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, vocab_size))
    for i in range(length):
        # appending the last predicted character to sequence
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        weights = model.predict(X[:, :i+1, :], batch_size=1)[0]
        if random:
            normalized = np.cumsum(weights, 1)[-1]
            normalized = normalized/normalized[-1]
            random = np.random.random()
            ix = [np.where(normalized>=random)[-1][0]]
        else:
            ix = np.argmax(weights, 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)

# method for preparing the training data
def load_data(data_dir, seq_length):
    data = open(data_dir, 'r').read()
    chars = list(set(data))
    VOCAB_SIZE = len(chars)

    print('Data length: {} characters'.format(len(data)))
    print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

    ix_to_char = {ix:char for ix, char in enumerate(chars)}
    char_to_ix = {char:ix for ix, char in enumerate(chars)}

    X = np.zeros((len(data)/seq_length, seq_length, VOCAB_SIZE))
    y = np.zeros((len(data)/seq_length, seq_length, VOCAB_SIZE))
    for i in range(0, len(data)/seq_length):
        X_sequence = data[i*seq_length:(i+1)*seq_length]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence]
        input_sequence = np.zeros((seq_length, VOCAB_SIZE))
        for j in range(seq_length):
            input_sequence[j][X_sequence_ix[j]] = 1.
            X[i] = input_sequence

        y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((seq_length, VOCAB_SIZE))
        for j in range(seq_length):
            target_sequence[j][y_sequence_ix[j]] = 1.
            y[i] = target_sequence
    return X, y, VOCAB_SIZE, ix_to_char, char_to_ix

def load_test_file(data_dir, char_to_ix, VOCAB_SIZE):
    data = open(data_dir, 'r').read()
    
    X = np.zeros((len(data), 1, VOCAB_SIZE))
    y = np.zeros((len(data), 1, VOCAB_SIZE))
    for i in range(0, len(data)):
        X_sequence = data[i:(i+1)]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence]
        input_sequence = np.zeros((1, VOCAB_SIZE))
        input_sequence[0][X_sequence_ix[0]] = 1.
        X[i] = input_sequence

        y_sequence = data[i+1:(i+1)+1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((1, VOCAB_SIZE))
        target_sequence[0][y_sequence_ix[0]] = 1.
        y[i] = target_sequence
    return X, y
        
    

def evaluate_loss(model, X, y):
    sess = K.get_session()
    preds = model.predict(X, batch_size=30)
    return tf.reduce_mean(losses.categorical_crossentropy(y, tf.convert_to_tensor(preds))).eval(session=sess)

def evaluate_loss_bad(model, excerpt, char_to_ix, vocab_size):
    # something isn't right here..
    X = np.zeros((1, len(excerpt), vocab_size))
    losses = []
    cum_loss = 0.
    ix = char_to_ix[excerpt[0]]
    X[0, 0, :][ix] = 1
    for i in range(1, len(excerpt)):
        char = excerpt[i]
        weights = model.predict(X[:, :i+1, :], batch_size=1)[0][-1]
#       print(sum(weights))
#        print(weights)
        ix = char_to_ix[char]
        for weight in weights:
            if weight < fuzz_factor:
                weight = fuzz_factor
            if weight > 1.-fuzz_factor:
                weight = 1.-fuzz_factor
        loss = -np.log(weights[ix])
        losses.append((char, weights[ix], loss))
        cum_loss += loss
        print('Char: ', char, ' Loss: ', loss,'  Loss so far: ', cum_loss/i)
        X[0, i, :][ix] = 1
    return losses

def make_lstm_model(VOCAB_SIZE, num_layers, num_hidden, dropout):
    model = Sequential()
    model.add(LSTM(num_hidden, input_shape=(None, VOCAB_SIZE), return_sequences=True, dropout=dropout))
    for i in range(num_layers - 1):
        model.add(LSTM(num_hidden, return_sequences=True,dropout=dropout))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    return model

def run_model(model, X, y, num_epochs=50, batch_size=30):
    early_stop1 = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=1e-3,
                              patience=2,
                              verbose=1, mode='auto')
    early_stop2 = keras.callbacks.EarlyStopping(monitor='loss',
                              min_delta=1e-3,
                              patience=2,
                              verbose=1, mode='auto')
    history = model.fit(X, y, batch_size=batch_size, callbacks=[early_stop1, early_stop2], validation_split=0.2, epochs=num_epochs, verbose=1)
    return model, history.history

def save_model(model, history, num_layers, num_hidden, dropout):
    string = 'layers_{}_hidden_{}_dropout_{}_epoch_{}'.format(num_layers, num_hidden, int(dropout*10), len(history['loss']))
    print('saving: ' + string)
    with open('history_'+string, 'wb') as file_pi:
        pickle.dump(history, file_pi)
    model.save_weights('weights_'+string+'.hdf5')