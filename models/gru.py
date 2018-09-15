from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class GRU(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim=256, units=1024):
    super().__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    if tf.test.is_gpu_available():
      print('Using CuDNNGRU')
      self.gru = tf.keras.layers.CuDNNGRU(self.units, 
                                          return_sequences=True, 
                                          return_state=True, 
                                          recurrent_initializer='glorot_uniform')
    else:
      self.gru = tf.keras.layers.GRU(self.units, 
                                     return_sequences=True, 
                                     return_state=True, 
                                     recurrent_activation='sigmoid', 
                                     recurrent_initializer='glorot_uniform')

    self.fc = tf.keras.layers.Dense(vocab_size)
        
  def call(self, x, hidden):
    x = self.embedding(x)

    # output shape == (batch_size, max_length, hidden_size) 
    # states shape == (batch_size, hidden_size)

    # states variable to preserve the state of the model
    # this will be used to pass at every step to the model while training
    output, states = self.gru(x, initial_state=hidden)


    # reshaping the output so that we can pass it to the Dense layer
    # after reshaping the shape is (batch_size * max_length, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # The dense layer will output predictions for every time_steps(max_length)
    # output shape after the dense layer == (max_length * batch_size, vocab_size)
    x = self.fc(output)

    return x, states
  
  # using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
  def loss_function(self, real, preds):
      return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)