import torch
from sklearn.metrics import f1_score, accuracy_score

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            preds.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    acc = accuracy_score(labels_list, preds)
    f1 = f1_score(labels_list, preds, average="macro")
    return acc, f1
