from datasources.full_text_datasource import FullTextDatasource
from datasources.sentences_datasource import SentencestDatasource


def load(text_path, ds_name='full-text', **kwargs):
    if ds_name == 'full-text':
        ds = FullTextDatasource(text_path, **kwargs)
    elif ds_name == 'sentences':
        ds = SentencestDatasource(text_path, **kwargs)
    else:
        raise ValueError('Unknown datasource: %s' % ds_name)
    
    ds.prepare_dataset()
    return ds