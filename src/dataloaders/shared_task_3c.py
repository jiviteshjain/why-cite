import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import cuda


class SharedTask3CDataset(Dataset):

    def __init__(self,
                 data,
                 tokenizer,
                 max_length,
                 is_test=False,
                 preprocess=None):
        self._data = data
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._is_test = is_test
        self._preprocess = preprocess

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        if self._preprocess:
            row = self._preprocess(self._data.iloc[index, :], self._is_test)
        else:
            row = self._data.iloc[index, :]

        tokenizer_args = {
            'add_special_tokens': False,
            'max_length': self._max_length,
            'pad_to_max_length': True,
            'return_token_type_ids': True,
            'truncation': True
        }

        citing_title = self._tokenizer.encode_plus(row['citing_title'],
                                                   text_pair=None,
                                                   **tokenizer_args)

        cited_title = self._tokenizer.encode_plus(row['cited_title'],
                                                  text_pair=None,
                                                  **tokenizer_args)

        citation_context = self._tokenizer.encode_plus(row['cited_title'],
                                                       text_pair=None,
                                                       **tokenizer_args)

        out = {
            'citing_title_ids':
                torch.tensor(citing_title['input_ids'], dtype=torch.long),
            'citing_title_mask':
                torch.tensor(citing_title['attention_mask'], dtype=torch.long),
            'cited_title_ids':
                torch.tensor(cited_title['input_ids'], dtype=torch.long),
            'cited_title_mask':
                torch.tensor(cited_title['attention_mask'], dtype=torch.long),
            'citation_context_ids':
                torch.tensor(citation_context['input_ids'], dtype=torch.long),
            'citation_context_mask':
                torch.tensor(citation_context['attention_mask'],
                             dtype=torch.long),
        }

        if not self._is_test:
            out['target'] = torch.tensor(row['target'], dtype=torch.long)

        return out


def clean_text(text):
    return ' '.join(text.strip().lower().split())


def preprocess(row, is_test=False):
    out = {
        'citing_title': clean_text(row['citing_title']),
        'cited_title': clean_text(row['cited_title']),
        'citation_context': clean_text(row['citation_context']),
    }

    if not is_test:
        out['target'] = int(row['target'])

    return out


def get_dataset(config, tokenizer, max_length):

    train_file_path = os.path.join(config.dataloaders.base_path,
                                   'shared_task_3c', 'train.csv')
    val_file_path = os.path.join(config.dataloaders.base_path, 'shared_task_3c',
                                 'validation.csv')

    train_data = pd.read_csv(
        train_file_path,
        sep=',',
        names=['citing_title', 'cited_title', 'citation_context', 'target'])
    val_data = pd.read_csv(
        val_file_path,
        sep=',',
        names=['citing_title', 'cited_title', 'citation_context', 'target'])

    train_dataset = SharedTask3CDataset(train_data,
                                        tokenizer,
                                        max_length,
                                        is_test=False,
                                        preprocess=preprocess)

    val_dataset = SharedTask3CDataset(val_data,
                                      tokenizer,
                                      max_length,
                                      is_test=False,
                                      preprocess=preprocess)

    # The last return value is the test split.
    return train_dataset, val_dataset, None
