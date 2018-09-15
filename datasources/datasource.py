import unidecode


class DataSource(object):
    def __init__(self, text_path, max_length=20, buffer_size=10000, batch_size=64, to_lower=False):
        self.text_path = text_path

        assert max_length > 0, 'Invalid max length %s' % max_length
        self.max_length = max_length

        assert buffer_size > 0, 'Invalid buffer size %s' % buffer_size
        self.buffer_size = buffer_size

        assert batch_size > 0, 'Invalid batch size %s' % batch_size
        self.batch_size = batch_size

        self.to_lower=to_lower

    def prepare_dataset(self):
        # Read text path
        self.text = unidecode.unidecode(open(self.text_path).read())
        # Unique contains all the unique characters in the file
        self.unique = sorted(set(self.text))
        # Creating a mapping from unique characters to indices
        self.char2idx = {u : i for i, u in enumerate(self.unique)}
        self.idx2char = {i : u for i, u in enumerate(self.unique)}
        # Vocab size
        self.vocab_size = len(self.unique)
        # Create dataset
        self._create_dataset()

    def _create_dataset(self):
        raise NotImplementedError()