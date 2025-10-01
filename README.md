# MRC_Transformers
This repo contains source code for our research work related to Machine Reading Comprehension Task for automated Question Answering 

- Datasets: Wiki-QA, SQuAD v2.0, NewsQA, Natural Questions (use HF `datasets` or local copies)
- Deep models: RNN / GRU / LSTM (PyTorch)
- Transformers: DistilBERT, RoBERTa, XLNet (HuggingFace)
- Vector DB: basic in-memory + optional FAISS
- Separate trainers, inference, evaluation (EM & F1)
- `server.py` to orchestrate training/inference

## Quickstart

1. Create a virtualenv and install requirements:
   ```bash
   pip install -r requirements.txt

### Example: 
Run a training run (DistilBERT) -- adapt args in server.py:
python src/server.py --mode train --model distilbert
