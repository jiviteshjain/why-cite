from transformers import AutoModel, AutoTokenizer
import torch


class MaheshwariEtAl(torch.nn.Module):

    def __init__(self, lang_model, dropout_probability, config, device):
        super().__init__()
        self._device = device

        self._lang_model = lang_model
        self._pre_classifier = torch.nn.Linear(768, 768)
        self._dropout_layer = torch.nn.Dropout(dropout_probability)
        self._classifier = torch.nn.Linear(768, 6)
        self._config = config

    def forward(self, data):

        if self._config.models.maheshwari_et_al.text.lower() != "extended":
            input_ids = data['citation_context_ids'].to(self._device,
                                                        dtype=torch.long)
            attention_mask = data['citation_context_mask'].to(self._device,
                                                            dtype=torch.long)
        else:
            input_ids = data['citation_extended_context_ids'].to(self._device,
                                                        dtype=torch.long)
            attention_mask = data['citation_extended_context_mask'].to(self._device,
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

    tokenizer.add_special_tokens({'additional_special_tokens' : ['@citation', '@CITATION']})
    lang_model.resize_token_embeddings(len(tokenizer))

    return MaheshwariEtAl(
        lang_model, config.models.maheshwari_et_al.dropout_probability, config,
        device), tokenizer, config.models.maheshwari_et_al.max_length
