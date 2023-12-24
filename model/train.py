from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from dataset.dataset import MethodNameDataset
from dataset.utils import get_methods_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score
import wandb

checkpoint = "Salesforce/codet5p-220m"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

train_methods, eval_methods = get_methods_split('../intellij-community')
train_dataset = MethodNameDataset(train_methods)
subset_size = int(0.01 * len(train_dataset))
train_subset = torch.utils.data.Subset(train_dataset, range(subset_size))

eval_dataset = MethodNameDataset(eval_methods)

train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

optimizer = AdamW(model.parameters(), lr=5e-5)


def train_and_evaluate(num_epochs, tokenizer, model, device, train_loader, val_loader, optimizer):
    wandb.init(project="MethodName")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        print(f"Epoch: {epoch}")
        actual_names_train = []
        predicted_names_train = []
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

            generated_ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                           max_new_tokens=7)
            predicted_name = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            actual_names_train.append([method_name])
            predicted_names_train.append(predicted_name)

        print(predicted_names_train)

        chencherry = SmoothingFunction()
        bleu_score_train = corpus_bleu(actual_names_train, predicted_names_train, smoothing_function=chencherry.method1)
        accuracy_train = accuracy_score([name[0] for name in actual_names_train], predicted_names_train)

        avg_train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_loss = 0
        actual_names = []
        predicted_names = []
        with torch.no_grad():
            for method_body, method_name in tqdm(val_loader):
                inputs = tokenizer(method_body, padding=True, truncation=True,
                                   return_tensors="pt").to(device)
                labels = tokenizer(method_name, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
                labels[labels == tokenizer.pad_token_id] = -100

                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                generated_ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                               max_new_tokens=7)
                predicted_name = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                actual_names_train.append([method_name])
                predicted_names_train.append(predicted_name)

        avg_val_loss = total_loss / len(val_loader)
        print(predicted_names)
        chencherry = SmoothingFunction()
        bleu_score_val = corpus_bleu(actual_names, predicted_names, smoothing_function=chencherry.method1)
        accuracy_val = accuracy_score([name[0] for name in actual_names], predicted_names)

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "train_bleu": bleu_score_train,
                   "val_bleu": bleu_score_val, "train_accuracy": accuracy_train, "val_accuracy": accuracy_val})


train_and_evaluate(1, tokenizer, model, device, train_loader, eval_loader, optimizer)
