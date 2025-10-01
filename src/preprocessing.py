# src/preprocessing.py
from transformers import AutoTokenizer
from typing import Dict, List

class Preprocessor:
    def __init__(self, model_name, max_len=384, doc_stride=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_len = max_len
        self.doc_stride = doc_stride

    def prepare_train_features(self, examples: Dict):
        # examples: {"question": [...], "context":[...], "answers":[{text,answer_start}]}
        tokenized = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_len,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        # Map start/end positions (approx) for SQuAD style
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            if len(answers["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                # find token indices
                token_start_index = 0
                token_end_index = len(input_ids) - 1
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= 0:
                    token_start_index += 1
                # locate tokens
                for idx, (s, e) in enumerate(offsets):
                    if s <= start_char < e:
                        token_start_index = idx
                    if s < end_char <= e:
                        token_end_index = idx
                start_positions.append(token_start_index)
                end_positions.append(token_end_index)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    def prepare_validation_features(self, examples: Dict):
        # similar but keep offsets for mapping back
        tokenized = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_len,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        return tokenized
