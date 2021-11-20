import nlpaug.augmenter.word as naw
import jsonlines
import copy
from tqdm import tqdm
import random

aug = naw.SynonymAug(aug_src='wordnet')
aug2 = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="substitute")

with jsonlines.open('./augmented_acl_processed_train2.jsonl', 'w') as write_file:
    with jsonlines.open('../data/acl-arc/processed_train.jsonl') as read_file:
        for obj in tqdm(read_file):
            write_file.write(obj)
            if obj['intent'] == "Future" or obj['intent'] == 'Extends' or obj['intent'] == 'Motivation':  
                temp_obj = copy.deepcopy(obj)
                temp_obj['text'] = aug.augment(obj['text'])
                temp_obj['extended_context'] = aug.augment(obj['extended_context'])
                write_file.write(temp_obj)
                temp_obj['text'] = aug2.augment(obj['text'])
                temp_obj['extended_context'] = aug2.augment(obj['extended_context'])
                write_file.write(temp_obj)
