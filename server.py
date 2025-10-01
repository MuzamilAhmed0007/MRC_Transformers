import argparse
from src.config import Config
from src.dataset_loader import DatasetLoader
from src.preprocessing import Preprocessor
from src.models.transformers_distilbert import DistilBERTQA
from src.trainer.trainer import Trainer
from transformers import AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.evaluation import evaluate_em_f1

def build_dataloaders(tokenized_train, tokenized_val, batch_size):
    # tokenized_* expected to be dict of tensors (pytorch tensors)
    def to_dataset(tok):
        import torch
        keys = ["input_ids","attention_mask","start_positions","end_positions"]
        tensors = [torch.tensor(tok[k]) for k in keys]
        return TensorDataset(*tensors)
    train_ds = to_dataset(tokenized_train)
    val_ds = to_dataset(tokenized_val)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size)

def main(args):
    cfg = Config()
    cfg.model_name_or_path = {
        "distilbert":"distilbert-base-uncased",
        "roberta":"deepset/roberta-base-squad2",
        "xlnet":"xlnet-base-cased"
    }.get(args.model, cfg.model_name_or_path)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DatasetLoader(cfg)
    hf_ds = loader.load()

    # NOTE: for speed, take small subset
    train = hf_ds["train"].select(range(min(200, len(hf_ds["train"]))))
    validation = hf_ds.get("validation", hf_ds["train"]).select(range(min(50, len(hf_ds.get("validation", hf_ds["train"])))))

    pre = Preprocessor(cfg.model_name_or_path, max_len=cfg.max_len, doc_stride=cfg.doc_stride)
    # build examples dict
    examples_train = {"question":[ex["question"] for ex in train],
                      "context":[ex["context"] for ex in train],
                      "answers":[{"text": ex["answers"]["text"], "answer_start": ex["answers"]["answer_start"]} for ex in train]}
    examples_val = {"question":[ex["question"] for ex in validation],
                    "context":[ex["context"] for ex in validation],
                    "answers":[{"text": ex["answers"]["text"], "answer_start": ex["answers"]["answer_start"]} for ex in validation]}

    tokenized_train = pre.prepare_train_features(examples_train)
    tokenized_val = pre.prepare_train_features(examples_val)

    # convert lists to tensors
    import torch
    for k in tokenized_train:
        tokenized_train[k] = torch.tensor(tokenized_train[k])
        tokenized_val[k] = torch.tensor(tokenized_val[k])

    train_loader, val_loader = build_dataloaders(tokenized_train, tokenized_val, cfg.batch_size)

    if args.model == "distilbert":
        model_wrapper = DistilBERTQA(cfg.model_name_or_path)
    else:
        # for brevity, fallback to DistilBERT
        model_wrapper = DistilBERTQA(cfg.model_name_or_path)
    model = model_wrapper.model
    device = cfg.device
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    trainer = Trainer(model, optimizer, device)

    best_val_loss = 1e9
    for epoch in range(cfg.epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc(start): {val_acc:.4f}")

    # quick inference on validation samples
    from src.inference import InferenceEngine
    inf = InferenceEngine(model, cfg.model_name_or_path, device=device)
    preds_and_gts = []
    for i in range(min(10, len(validation))):
        q = validation[i]["question"]
        c = validation[i]["context"]
        pred = inf.predict(q, c)
        gt = validation[i]["answers"]["text"][0] if validation[i]["answers"]["text"] else ""
        preds_and_gts.append((pred, gt))
    metrics = evaluate_em_f1(preds_and_gts)
    print("EM/F1:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","eval","infer"], default="train")
    parser.add_argument("--model", choices=["distilbert","roberta","xlnet"], default="distilbert")
    args = parser.parse_args()
    main(args)
