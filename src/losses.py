# TODO(jiviteshjain): This is not compatible with huggingface
# AutoModelsForTask, because they include the loss as well.
# Decide if we want to use their inbuilt losses.

import torch


def weighted_cross_entropy(config, device):
    weights = torch.tensor(config.losses.weighted_cross_entropy.weights,
                           dtype=torch.float32).to(device)
    return torch.nn.CrossEntropyLoss(weight=weights)


def multi_task_weighted_cross_entropy(config, device):
    num_tasks = config.losses.multi_task_weighted_cross_entropy.num_tasks
    weights = torch.tensor(
        config.losses.multi_task_weighted_cross_entropy.weights,
        dtype=torch.float32).to(device)
    importance = torch.tensor(
        config.losses.multi_task_weighted_cross_entropy.importance,
        dtype=torch.float32).to(device)

    losses = [
        torch.nn.CrossEntropyLoss(weight=weights[i]) for i in range(num_tasks)
    ]

    def compute(outputs, targets):
        loss = torch.tensor(0, dtype=torch.float32, device=device)

        for i in range(num_tasks):
            loss += (importance[i] * losses[i](outputs[i], targets[..., i]))

        return loss

    return compute


def losses(loss, config, device):
    getters = {
        'weighted_cross_entropy': weighted_cross_entropy,
        'multi_task_weighted_cross_entropy': multi_task_weighted_cross_entropy,
    }

    return getters[loss](config, device)
