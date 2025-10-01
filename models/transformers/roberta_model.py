import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

class RobertaClassifier(nn.Module):
    def __init__(self, num_labels=4, dropout=0.3):
        super(RobertaClassifier, self).__init__()
        self.config = RobertaConfig.from_pretrained("roberta-base", num_labels=num_labels)
        self.roberta = RobertaModel.from_pretrained("roberta-base", config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

def get_tokenizer():
    return RobertaTokenizer.from_pretrained("roberta-base")

def get_model(num_labels=4):
    return RobertaClassifier(num_labels=num_labels)
