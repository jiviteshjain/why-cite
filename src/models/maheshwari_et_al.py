from transformers import AutoModel, AutoTokenizer
import torch
from omegaconf.errors import ConfigKeyError


class MaheshwariEtAl(torch.nn.Module):

    def __init__(self, lang_model, dropout_probability, device):
        super().__init__()
        self._device = device

        print(self._device, "is device")

        self._lang_model = lang_model.to(device)
        self._pre_classifier = torch.nn.Linear(768, 768).to(device)
        self._dropout_layer = torch.nn.Dropout(dropout_probability).to(device)
        self._classifier = torch.nn.Linear(768, 6).to(device)

    def forward(self, data):

        input_ids = data['citation_context_ids'].to(self._device,
                                                    dtype=torch.long)
        attention_mask = data['citation_context_mask'].to(self._device,
                                                          dtype=torch.long)

        embeddings = self._lang_model(input_ids=input_ids,
                                      attention_mask=attention_mask)[0]
        hidden_state = embeddings[:, 0]
        hidden_state = self._pre_classifier(hidden_state)
        hidden_state = torch.nn.ReLU()(hidden_state)
        hidden_state = self._dropout_layer(hidden_state)
        return self._classifier(hidden_state)


def get_model(config, device):
    lang_model = AutoModel.from_pretrained(
        config.models.maheshwari_et_al.pretrained_identifier)
    tokenizer = AutoTokenizer.from_pretrained(
        config.models.maheshwari_et_al.pretrained_identifier)

    try:
        special_tokens = [str(x) for x in config.dataloaders[
            config.training.dataset_in_use].special_tokens]
        
        tokenizer.add_special_tokens(
            {'additional_special_tokens': special_tokens})
        lang_model.resize_token_embeddings(len(tokenizer))
    
    except ConfigKeyError:
        pass

    return MaheshwariEtAl(
        lang_model, config.models.maheshwari_et_al.dropout_probability,
        device), tokenizer, config.models.maheshwari_et_al.max_length
