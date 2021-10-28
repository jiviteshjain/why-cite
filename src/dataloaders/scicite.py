import os

import pandas as pd
import numpy as np
import torch
import jsonlines

from torch.utils.data import DataLoader, Dataset
from torch import cuda

INTENT_CATEGORY_MAP = {
    'background': 0,
    'method': 1,
    'result': 2
}

SECTION_CATEGORY_MAP = {
    'introduction': 0,
    'results': 1,
    'discussion': 2,
    'methods': 3,
    'experiments': 4,
    'related work': 5,
    'conclusion': 6,
    'unknown': 7
}


class SciCiteDataset(Dataset):

    def __init__(self,
                 data,
                 tokenizer,
                 max_length,
                 is_test=False,
                 section_in_target=False,
                 preprocess=None):
        self._data = data
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._is_test = is_test
        self._section_in_target = section_in_target
        self._preprocess = preprocess

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        if self._preprocess:
            row = self._preprocess(self._data[index], self._is_test)
        else:
            row = self._data[index]
            row['target'] = INTENT_CATEGORY_MAP[row['label']]
            section_name = row['sectionName'] if row[
                'sectionName'] is not None else 'unknown'
            row['section_name'] = SECTION_CATEGORY_MAP[section_name]

        tokenizer_args = {
            'add_special_tokens': True,
            'max_length': self._max_length,
            'padding': 'max_length',
            'return_token_type_ids': True,
            'truncation': True
        }

        citing_title = self._tokenizer.encode_plus('',
                                                   text_pair=None,
                                                   **tokenizer_args)

        cited_title = self._tokenizer.encode_plus('',
                                                  text_pair=None,
                                                  **tokenizer_args)

        citation_context = self._tokenizer.encode_plus(row['string'],
                                                           text_pair=None,
                                                           **tokenizer_args)
       
        # Uncomment this to add section name
        # section_name = row['section_name']

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

        if not self._section_in_target:
            out['section'] = torch.tensor(row['section_name'])

        if not self._is_test:
            if not self._section_in_target:
                out['target'] = torch.tensor(row['target'], dtype=torch.long)
            else:
                out['target'] = torch.tensor(
                    [row['target'], row['section_name']], dtype=torch.long)

        return out


def clean_text(text):
    return ' '.join(text.strip().lower().split())


def preprocess(row, is_test=False):

    section_name = row['sectionName'] if row[
        'sectionName'] is not None else 'unknown'
    section_name = SECTION_CATEGORY_MAP[section_name]

    out = {
        'string': clean_text(row['string']),
        'extended_context': clean_text(row['string']),
        'section_name': section_name,
    }

    if not is_test:
        out['target'] = INTENT_CATEGORY_MAP[row['label']]

    return out


def load_jsonl(path):
    data = jsonlines.open(path)
    return list(data)


def get_dataset(config, tokenizer, max_length):

    train_file_path = os.path.join(config.dataloaders.base_path, 'scicite',
                                   'processed_train.jsonl')
    val_file_path = os.path.join(config.dataloaders.base_path, 'scicite',
                                 'processed_dev.jsonl')
    test_file_path = os.path.join(config.dataloaders.base_path, 'scicite',
                                  'processed_test.jsonl')

    train_data = load_jsonl(train_file_path)
    val_data = load_jsonl(val_file_path)
    test_data = load_jsonl(test_file_path)

    print(config.dataloaders)

    train_dataset = SciCiteDataset(
        train_data,
        tokenizer,
        max_length,
        is_test=False,
        section_in_target=config.dataloaders.scicite.section_in_target,
        preprocess=preprocess)

    val_dataset = SciCiteDataset(
        val_data,
        tokenizer,
        max_length,
        is_test=False,
        section_in_target=config.dataloaders.scicite.section_in_target,
        preprocess=preprocess)

    test_dataset = SciCiteDataset(
        test_data,
        tokenizer,
        max_length,
        is_test=True,
        section_in_target=config.dataloaders.scicite.section_in_target,
        preprocess=preprocess)

    # The last return value is the test split.
    return train_dataset, val_dataset, test_dataset
