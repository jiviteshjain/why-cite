# This thing should be called 'datasets' but huggingface
# already stole that.

#TODO(jiviteshjain): Add support for ClassLabels.

import os

from . import shared_task_3c, acl_arc, scicite


def dataloaders(dataset, config, tokenizer, max_length):
    getters = {
        'shared_task_3c': shared_task_3c.get_dataset,
        'acl_arc': acl_arc.get_dataset,
        'scicite': scicite.get_dataset,
    }

    return getters[dataset](config, tokenizer, max_length)