from . import maheshwari_et_al
from . import two_task_transformer


def models(model, config, device):
    getters = {
        'maheshwari_et_al': maheshwari_et_al.get_model,
        'two_task_transformer': two_task_transformer.get_model
    }

    return getters[model](config, device)