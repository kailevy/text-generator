from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse
from RNN_utils import *

# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='./data/test.txt')
ap.add_argument('-batch_size', type=int, default=50)
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=50)
ap.add_argument('-hidden_dim', type=int, default=500)
ap.add_argument('-generate_length', type=int, default=500)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
ap.add_argument('-start', default='')
ap.add_argument('-randgen', default='')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']

start = args['start']
randgen = args['randgen']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

# Creating training data
X, y, VOCAB_SIZE, ix_to_char, char_to_ix = load_data(DATA_DIR, SEQ_LENGTH)

# Creating and compiling the Network
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
  model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# Generate some sample before training to know how bad it is!
#print("\n\nGenerating untrained text...\n\n")
#generate_text(model, args['generate_length'], VOCAB_SIZE, ix_to_char, -1, 0)

if not WEIGHTS == '':
  model.load_weights(WEIGHTS)
  nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
else:
  nb_epoch = 0

# Training if there is no trained weights specified
if args['mode'] == 'train' or WEIGHTS == '':
  while True:
    print('\n\nEpoch: {}\n'.format(nb_epoch))
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
    nb_epoch += 1
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char, -1, 0)
    if nb_epoch % 10 == 0:
      model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch))

# Else, loading the trained weights and performing generation only
elif WEIGHTS == '':
  # Loading the trained weights
  model.load_weights(WEIGHTS)
  generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char, -1, 0)
  print('\n\n')
# elif not start == '':
#   if not randgen == '':
#     print("\n\nGenerating random text with start..\n\n")
#     start_ix = [char_to_ix[s] for s in start]
#     generate
#   else:
#     print("\n\nGenerating text with start...\n\n")
#     start_ix = [char_to_ix[s] for s in start]
#     generate_with_start(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char, start_ix)
# elif not randgen == '':
#   print("\n\nGenerating random text..\n\n")
#   generate_random(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
else:
  print("\n\nGenerating trained text...\n\n")
  if not start == '':
    print("with start...")
    start_ix = [char_to_ix[s] for s in start]
  else:
    start_ix = -1
  if not randgen == '':
    print("randomly..")
    randgen = 1
  else:
    randgen = 0
  generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char, start_ix, randgen)
