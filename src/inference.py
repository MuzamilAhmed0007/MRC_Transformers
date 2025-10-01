# src/inference.py
import torch
from transformers import AutoTokenizer
import numpy as np

class InferenceEngine:
    def __init__(self, hf_model, tokenizer_name, device="cpu"):
        self.model = hf_model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.model.to(device)

    def predict(self, question, context, max_ans_len=30):
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        outputs = self.model(**inputs)
        start_logits = outputs.start_logits.detach().cpu().numpy()[0]
        end_logits = outputs.end_logits.detach().cpu().numpy()[0]
        # naive span pick
        start_idx = np.argmax(start_logits)
        end_idx = np.argmax(end_logits)
        if end_idx < start_idx:
            end_idx = start_idx
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx+1]))
        return answer
