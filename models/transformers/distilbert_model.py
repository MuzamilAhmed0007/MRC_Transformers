import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig

class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels=4, dropout=0.3):
        super(DistilBertClassifier, self).__init__()
        # Load configuration
        self.config = DistilBertConfig.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        # Pre-trained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", config=self.config
        )
        # Dropout + Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass:
        - input_ids: [batch_size, seq_len]
        - attention_mask: [batch_size, seq_len]
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_state = outputs.last_hidden_state   # [batch_size, seq_len, hidden_dim]
        pooled_output = hidden_state[:, 0]         # take [CLS]-like embedding
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

def get_tokenizer():
    """Load DistilBERT tokenizer"""
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def get_model(num_labels=4):
    """Return classifier model"""
    return DistilBertClassifier(num_labels=num_labels)
