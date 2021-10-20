from . import maheshwari_et_al


def models(model, config, device):
    getters = {
        'maheshwari_et_al': maheshwari_et_al.get_model,
    }

    return getters[model](config, device)