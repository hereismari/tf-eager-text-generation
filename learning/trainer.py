import time
import numpy as np
import string
import os
import random

import tensorflow as tf

from managers.checkpoint import CheckpointManager

class Trainer(object):
    def __init__(self, datasource, model, optimizer, checkpoint_dir, checkpoint_prefix):
        self.ds = datasource
        self.model = model
        self.optimizer = optimizer
        self.checkpoint = CheckpointManager(checkpoint_dir, checkpoint_prefix,
                                            optimizer, model)
    
    def restore_last_checkpoint(self):
        self.checkpoint.restore_last()

    def train(self, epochs=10, verbose=True, num_char_generate=30, start_string=None, temperature=1.0):
        for epoch in range(epochs):
            start = time.time()
            # Initializing the hidden state at the start of every epoch
            hidden = self.model.reset_states()
            loss = 0
            for (batch, (inp, target)) in enumerate(self.ds.dataset):
                with tf.GradientTape() as tape:
                    # feeding the hidden state back into the model
                    # This is the interesting step
                    predictions, hidden = self.model(inp, hidden)
                    
                    # reshaping the target because that's how the 
                    # loss function expects it
                    target = tf.reshape(target, (-1,))
                    loss = self.model.loss_function(target, predictions)
                    
                grads = tape.gradient(loss, self.model.variables)
                self.optimizer.apply_gradients(zip(grads, self.model.variables))

                if batch % 100 == 0 and verbose:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                                  batch,
                                                                  loss))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save()

            print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
            print ('Time taken for epoch {}: {} sec\n'.format(epoch+1, time.time() - start))
            print (self.sample(num_char_generate, start_string, temperature))


    def sample(self, num_char_generate=30, start_string=None, temperature=1.0):
        '''
            Generates text similar to the training text.

            Args:
                num_char_generate: number of characters to geneterate.
                start_string: input string to start generating text,
                    if None it will be one random valid character.
                temperature: low temperature results in more predictable text,
                    higher temperatures results in more surprising text.
            Returns:
                A string composed by the start_string + RNN generated text.
        '''
        # if start_string is none we choose a random starting char
        if start_string is None:
            start_string = random.choice(self.ds.unique)
        else:
            for char in start_string:
                if char not in self.ds.char2idx:
                    print ('Invalid start string')
                    return

        # Converting our start string to numbers(vectorizing!) 
        input_eval = [self.ds.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # empty string to store our results
        text_generated = ''

        # hidden state shape == (batch_size, number of rnn units); here batch size == 1
        hidden = [tf.zeros((1, self.model.units))]
        for i in range(num_char_generate):
            predictions, hidden = self.model(input_eval, hidden)

            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()

            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated += self.ds.idx2char[predicted_id]

        return ((start_string + text_generated).replace('<pad>', ''))
