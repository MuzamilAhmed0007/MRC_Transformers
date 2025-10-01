import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from data_loader import train_dataset, val_dataset   # your preprocessed datasets
from models import bert_model, roberta_model, xlnet_model
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose model
tokenizer = bert_model.get_tokenizer()
model = bert_model.get_model(num_labels=4).to(device)

# tokenizer = roberta_model.get_tokenizer()
# model = roberta_model.get_model(num_labels=4).to(device)

# tokenizer = xlnet_model.get_tokenizer()
# model = xlnet_model.get_model(num_labels=4).to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

def train(epochs=3):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")
        acc, f1 = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    train(epochs=3)
