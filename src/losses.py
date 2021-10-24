# TODO(jiviteshjain): This is not compatible with huggingface
# AutoModelsForTask, because they include the loss as well.
# Decide if we want to use their inbuilt losses.

import torch


def weighted_cross_entropy(config, device):
    weights = torch.FloatTensor(config.losses.weighted_cross_entropy.weights).to(device)
    return torch.nn.CrossEntropyLoss(weight=weights)


def losses(loss, config, device):
    getters = {
        'weighted_cross_entropy': weighted_cross_entropy,
    }

    return getters[loss](config, device)
