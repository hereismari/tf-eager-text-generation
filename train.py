import time
import numpy as np
import random
import string
import os

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
# Note: Once you enable eager execution, it cannot be disabled. 
tf.enable_eager_execution()

import datasources
import models

def main():
    ds = datasources.load('colors.txt', ds_name='sentences', max_length=40)
    model = models.load(ds.vocab_size)
    optimizer = tf.train.AdamOptimizer()

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model)

    # Training step
    EPOCHS = 20

    for epoch in range(EPOCHS):
        start = time.time()
        
        # initializing the hidden state at the start of every epoch
        hidden = model.reset_states()
        
        loss = 0
        for (batch, (inp, target)) in enumerate(ds.dataset):
            with tf.GradientTape() as tape:
                # feeding the hidden state back into the model
                # This is the interesting step
                predictions, hidden = model(inp, hidden)
                
                # reshaping the target because that's how the 
                # loss function expects it
                target = tf.reshape(target, (-1,))
                loss = model.loss_function(target, predictions)
                
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                                batch,
                                                                loss))
        # saving (checkpoint) the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        
        # Evaluation step(generating text using the model learned)
        for i in range(5):
            # number of characters to generate
            num_generate = np.random.randint(20, 40)

            # You can change the start string to experiment

            start_string = random.choice(string.ascii_lowercase)
            # converting our start string to numbers(vectorizing!) 
            input_eval = [ds.char2idx[s] for s in start_string]
            input_eval = tf.expand_dims(input_eval, 0)

            # empty string to store our results
            text_generated = ''

            # low temperatures results in more predictable text.
            # higher temperatures results in more surprising text
            # experiment to find the best setting
            temperature = 0.8

            units = 1024
            # hidden state shape == (batch_size, number of rnn units); here batch size == 1
            hidden = [tf.zeros((1, units))]
            for i in range(num_generate):
                predictions, hidden = model(input_eval, hidden)

                # using a multinomial distribution to predict the word returned by the model
                predictions = predictions / temperature
                predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()

                # We pass the predicted word as the next input to the model
                # along with the previous hidden state
                input_eval = tf.expand_dims([predicted_id], 0)

                text_generated += ds.idx2char[predicted_id]

            # try_plot(start_string + text_generated)
            print(start_string + text_generated)


if __name__ == '__main__':
    main()