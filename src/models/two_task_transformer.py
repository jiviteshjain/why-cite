from transformers import AutoModel, AutoTokenizer
import torch
from omegaconf.errors import ConfigKeyError


class TwoTaskTransformer(torch.nn.Module):

    def __init__(self, lang_model, intent_dropout_probability,
                 section_dropout_probability, device):
        super().__init__()
        self._device = device

        self._lang_model = lang_model.to(device)

        self._heads = {
            'intent': {
                'pre_classifier':
                    torch.nn.Linear(768, 768).to(device),
                'dropout_layer':
                    torch.nn.Dropout(intent_dropout_probability).to(device),
                'classifier':
                    torch.nn.Linear(768, 6).to(device),
            },
            'section': {
                'pre_classifier':
                    torch.nn.Linear(768, 768).to(device),
                'dropout_layer':
                    torch.nn.Dropout(section_dropout_probability).to(device),
                'classifier':
                    torch.nn.Linear(768, 6).to(device),
            }
        }

    def _pass_through_head(self, task, hidden_state):
        intermediate = self._heads[task]['pre_classifier'](hidden_state)
        intermediate = torch.nn.ReLU()(intermediate)
        intermediate = self._heads[task]['dropout_layer'](intermediate)
        return self._heads[task]['classifier'](intermediate)

    def forward(self, data):
        input_ids = data['citation_context_ids'].to(self._device,
                                                    dtype=torch.long)
        attention_mask = data['citation_context_mask'].to(self._device,
                                                          dtype=torch.long)

        embeddings = self._lang_model(input_ids=input_ids,
                                      attention_mask=attention_mask)[0]
        hidden_state = embeddings[:, 0]

        return [
            self._pass_through_head('intent', hidden_state),
            self._pass_through_head('section', hidden_state)
        ]


def get_model(config, device):
    lang_model = AutoModel.from_pretrained(
        config.models.two_task_transformer.pretrained_identifier)
    tokenizer = AutoTokenizer.from_pretrained(
        config.models.two_task_transformer.pretrained_identifier)

    try:
        special_tokens = [str(x) for x in config.dataloaders[
            config.training.dataset_in_use].special_tokens]
        
        tokenizer.add_special_tokens(
            {'additional_special_tokens': special_tokens})
        lang_model.resize_token_embeddings(len(tokenizer))
    
    except ConfigKeyError:
        pass

    return TwoTaskTransformer(
        lang_model,
        config.models.two_task_transformer.intent_dropout_probability,
        config.models.two_task_transformer.section_dropout_probability,
        device), tokenizer, config.models.two_task_transformer.max_length
