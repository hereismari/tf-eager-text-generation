import os
import tensorflow as tf


class CheckpointManager(object):

    def __init__(self, checkpoint_dir, checkpoint_prefix, optimizer, model):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, checkpoint_prefix)
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    def save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)