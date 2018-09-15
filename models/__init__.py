from models.gru import GRU


def load(vocab_size, model_name='gru', **kwargs):
    if model_name == 'gru':
        model = GRU(vocab_size, **kwargs)
    else:
        raise ValueError('Unknown model: %s' % model_name)
    
    return model