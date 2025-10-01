# src/dataset_loader.py
from datasets import load_dataset
from typing import Dict, Any
import os

class DatasetLoader:
    def __init__(self, config):
        self.config = config

    def load(self):
        name = self.config.dataset_name.lower()
        if name in ["squad", "squad_v2", "squad_v2.0"]:
            return load_dataset("squad_v2")
        elif name == "wiki-qa" or name == "wikiqa":
            # fallback: example using HF dataset name 'wiki_qa' if available; otherwise expect local files
            try:
                return load_dataset("wiki_qa")
            except Exception:
                raise NotImplementedError("Please provide local Wiki-QA preprocessing or use HF name.")
        elif name == "newsqa":
            try:
                return load_dataset("newsqa")
            except Exception:
                raise NotImplementedError("Please provide local NewsQA preprocessing.")
        elif name in ["natural_questions", "nq"]:
            try:
                return load_dataset("natural_questions")
            except Exception:
                # HF dataset for Natural Questions might be 'natural_questions' or 'nq_open'
                return load_dataset("natural_questions", "short")
        else:
            # try local JSON/CSV file
            path = os.path.join(self.config.data_dir, self.config.dataset_name)
            if os.path.exists(path):
                return load_dataset('json', data_files={'train': path})
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")

    # simple helper to convert to question-context-answer triples (SQuAD-style)
    @staticmethod
    def squad_style_examples(hf_dataset):
        # returns generator/dict for sample usage
        for split in hf_dataset:
            for ex in hf_dataset[split]:
                yield ex
