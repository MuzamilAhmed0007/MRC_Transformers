import torch
from torch import nn
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

class XLNetClassifier(nn.Module):
    def __init__(self, num_labels=4, dropout=0.3):
        super(XLNetClassifier, self).__init__()
        self.config = XLNetConfig.from_pretrained("xlnet-base-cased", num_labels=num_labels)
        self.xlnet = XLNetModel.from_pretrained("xlnet-base-cased", config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, -1, :]  # XLNet uses last token as summary
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

def get_tokenizer():
    return XLNetTokenizer.from_pretrained("xlnet-base-cased")

def get_model(num_labels=4):
    return XLNetClassifier(num_labels=num_labels)
