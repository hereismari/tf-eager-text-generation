import argparse
parser = argparse.ArgumentParser()

# Datasource
parser.add_argument('--text-path', type=str, required=True)
parser.add_argument('--text-type', type=str, default='full-text', choices=['full-text', 'sentences'])
parser.add_argument('--max-length', type=int, default=100)
parser.add_argument('--buffer-size', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=64)

parser.add_argument('--to-lower', dest='to_lower', action='store_true')
parser.set_defaults(to_lower=False)

# Model
parser.add_argument('--model', type=str, default='gru', choices=['gru'])
parser.add_argument('--embedding-dim', type=int, default=256)
parser.add_argument('--units', type=int, default=1024)

# Checkpoint
parser.add_argument('--checkpoint-dir', type=str, default='pretrained_models')

# Trainer
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)

# Sample
parser.add_argument('--num-char-generate', type=int, default=30)
parser.add_argument('--start-string', type=str, default=None)
parser.add_argument('--temperature', type=float, default=1.0)