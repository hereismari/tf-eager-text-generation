# -*- coding: utf-8 -*-
import os

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
# Note: Once you enable eager execution, it cannot be disabled. 
tf.enable_eager_execution()

import datasources
import models
from learning.trainer import Trainer

from utils.args import parser


def main():
    args = parser.parse_args()
    ds = datasources.load(args.text_path, ds_name=args.text_type,
                        max_length=args.max_length,
                        buffer_size=args.buffer_size,
                        batch_size=args.batch_size,
                        to_lower=args.to_lower)

    model = models.load(ds.vocab_size, embedding_dim=args.embedding_dim, units=args.units)

    checkpoint_prefix = os.path.splitext(os.path.basename(args.text_path))[0]
    trainer = Trainer(datasource=ds, model=model, optimizer=tf.train.AdamOptimizer(),
                      checkpoint_dir=args.checkpoint_dir, checkpoint_prefix=checkpoint_prefix)

    trainer.restore_last_checkpoint()

    print('Insert separated by spaces:')
    print('- number of samples that should be generated')
    print('- size of text to be generated')
    print('- the start string of the text')
    print('- temperature (higher = more creative, lower=more predictable')
    print('Example of input: 10 100 test 1.2')
    

    while True:
        repeat, num_char_generate, start_string, temperature = input().split()
        repeat = int(repeat)
        temperature = float(temperature)
        num_char_generate = int(num_char_generate)
        for i in range(repeat):
            generated_text = trainer.sample(num_char_generate=num_char_generate, start_string=start_string, temperature=temperature)
            if args.text_type == 'sentences' and generated_text in ds.text:
                print ('Not so creative this exact sample of text is in the train file.')
            print (generated_text)
            print ('-' * 30)
            

if __name__ == '__main__':
    main()