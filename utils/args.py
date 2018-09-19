import argparse
parser = argparse.ArgumentParser()

# Datasource
parser.add_argument('--text-path', type=str, required=True, help='Path to text file.')
parser.add_argument('--text-type', type=str, default='full-text', choices=['full-text', 'sentences'],
                    help='Type of text\nsenteces = consider each line of file as a piece of text.\n'
                         'full-text = consider text file as a single long text.')
parser.add_argument('--max-length', type=int, default=100, help='Max length of a piece of text used to train model.')
parser.add_argument('--buffer-size', type=int, default=10000, help='Buffer size to shuffle dataset.')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')

parser.add_argument('--to-lower', dest='to_lower', action='store_true', help='Convert all chars to lower case.')
parser.set_defaults(to_lower=False)

# Model
parser.add_argument('--model', type=str, default='gru', choices=['gru'], help='RNN cell, for now only GRU is available.')
parser.add_argument('--embedding-dim', type=int, default=256, help='Embedding dimension in model.')
parser.add_argument('--units', type=int, default=1024, help='RNN units.')

# Checkpoint
parser.add_argument('--checkpoint-dir', type=str, default='pretrained_models', help='Path to save models checkpoints.')

# Trainer
parser.add_argument('--epochs', type=int, default=40, help='Train epochs.')
parser.add_argument('--verbose', dest='verbose', action='store_true', help='Generate text as it train.')
parser.set_defaults(verbose=False)

# Sample
parser.add_argument('--num-char-generate', type=int, default=30, help='Number of chars to generate as it train.')
parser.add_argument('--start-string', type=str, default=None, help='Start string to generate text. If None a random char will be used instead.')
parser.add_argument('--temperature', type=float, default=1.0, help='Higher = more creative text, lower = boring text.')