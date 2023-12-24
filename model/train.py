from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from dataset.dataset import MethodNameDataset
from dataset.utils import get_methods_split

checkpoint = "Salesforce/codet5p-770m"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

train_methods, eval_methods = get_methods_split('../intellij-community')
train_dataset = MethodNameDataset(train_methods)
eval_dataset = MethodNameDataset(eval_methods)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

optimizer = AdamW(model.parameters(), lr=5e-5)


def evaluate(tokenizer, model, device, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _, (method_body, method_name) in enumerate(loader):
            inputs = tokenizer(method_body, padding=True, truncation=True,
                               return_tensors="pt").to(device)
            labels = tokenizer(method_name, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(loader)


def train_and_evaluate(num_epochs, tokenizer, model, device, train_loader, val_loader, optimizer):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        print(f"Epoch: {epoch}")
        for (method_body, method_name) in tqdm(train_loader):
            inputs = tokenizer(method_body, padding=True, truncation=True, return_tensors="pt").to(device)
            labels = tokenizer(method_name, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"Epoch: {epoch}, Loss: {loss.item()}")

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate(tokenizer, model, device, val_loader)

        print(f"Epoch: {epoch}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")


train_and_evaluate(5, tokenizer, model, device, train_loader, eval_loader, optimizer)
