import tensorflow as tf
import unidecode


from datasources.datasource import DataSource


class SentencestDatasource(DataSource):
    def __init__(self, text_path, max_length=20, buffer_size=10000, batch_size=64, to_lower=False):
        super().__init__(text_path, max_length=max_length, buffer_size=buffer_size, batch_size=batch_size, to_lower=to_lower)
    
    def prepare_dataset(self):
        # Read text path
        self.text = open(self.text_path).readlines()
        
        # Unique contains all the unique characters in the file
        self.chars = set()
        for sentence in self.text:
            for char in sentence:
                self.chars.add(char)
        self.unique = sorted(set(self.chars))

        # Creating a mapping from unique characters to indices
        self.char2idx = {u : i for i, u in enumerate(self.unique)}
        self.idx2char = {i : u for i, u in enumerate(self.unique)}

        # Vocab size
        self.vocab_size = len(self.unique) + 1  # pad
        
        # Create dataset
        self._create_dataset()
    
    def _pad_text(self):
        text_int = []
        for sentence in self.text:
            pad = [] 
            for c in sentence:
                pad.append(self.char2idx[c])
            
            assert [self.char2idx[c] for c in sentence] == pad
            
            if len(pad) < self.max_length:
                pad =  pad + [self.char2idx['<pad>']] * (self.max_length - len(pad))
            elif len(pad) > self.max_length:
                continue
            assert len(pad) == self.max_length
            text_int.append(pad)
        return text_int


    def _create_dataset(self):
        self.idx2char[len(self.idx2char)] = '<pad>'
        self.char2idx['<pad>'] = len(self.char2idx)

        self.text_int = self._pad_text()
       
        self.input_text = []
        self.target_text = []

        for i in range(len(self.text_int)):
            inps = self.text_int[i][:self.max_length-1]
            targ = self.text_int[i][1:self.max_length]
            self.input_text.append(inps)
            self.target_text.append(targ)

        self.dataset = tf.data.Dataset.from_tensor_slices((self.input_text, self.target_text)).shuffle(self.buffer_size)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
