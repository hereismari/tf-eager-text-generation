import tensorflow as tf
from datasources.datasource import DataSource


class FullTextDatasource(DataSource):
    def __init__(self, text_path, max_length=20, buffer_size=10000, batch_size=64, to_lower=False):
        super().__init__(text_path, max_length=max_length, buffer_size=buffer_size, batch_size=batch_size, to_lower=to_lower)

    def _create_dataset(self):
        self.input_text = []
        self.target_text = []

        for f in range(0, len(self.text)-self.max_length, self.max_length):
            inps = self.text[f : f + self.max_length]
            targ = self.text[f + 1 : f + 1 + self.max_length]

            self.input_text.append([self.char2idx[i] for i in inps])
            self.target_text.append([self.char2idx[t] for t in targ])

        self.dataset = tf.data.Dataset.from_tensor_slices((self.input_text, self.target_text)).shuffle(self.buffer_size)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
