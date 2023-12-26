from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from dataset.dataset import MethodNameDataset
from dataset.utils import get_methods_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score
import wandb
import Levenshtein as lev

checkpoint = "Salesforce/codet5p-220m"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

train_methods, eval_methods = get_methods_split('../intellij-community')
train_dataset = MethodNameDataset(train_methods)
subset_size = int(len(train_dataset))

train_subset = torch.utils.data.Subset(train_dataset, range(subset_size))

eval_dataset = MethodNameDataset(eval_methods)
subset_size_eval = int(len(eval_dataset))
eval_dataset = torch.utils.data.Subset(eval_dataset, range(subset_size_eval))

train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

optimizer = Adam(model.parameters(), lr=1e-6)


def iou_score(pred: str, truth: str):
    intersection = set(pred).intersection(set(truth))
    union = set(pred).union(set(truth))
    return len(intersection) / len(union)


def train_and_evaluate(num_epochs, tokenizer, model, device, train_loader, val_loader, optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=2, verbose=True)
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
            # print(outputs)
            loss = outputs.loss
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            generated_ids = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                           max_new_tokens=3)
            predicted_names = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

            actual_names_train += list(method_name)
            predicted_names_train += predicted_names

        chencherry = SmoothingFunction()
        bleu_score_train = corpus_bleu(actual_names_train, predicted_names_train, smoothing_function=chencherry.method1)
        accuracy_train = accuracy_score(actual_names_train, predicted_names_train)
        avg_train_loss = total_train_loss / len(train_loader)

        # avg_rogue_l_train = calculate_rouge_l(predicted_names_train, actual_names_train)
        iou_scores_train = [iou_score(pred, truth) for pred, truth in zip(predicted_names_train, actual_names_train)]
        avg_iou_train = sum(iou_scores_train) / len(iou_scores_train)
        lev_dists_train = [lev.distance(pred, truth) for pred, truth in zip(predicted_names_train, actual_names_train)]
        avg_lev_dist_train = sum(lev_dists_train) / len(lev_dists_train)

        model.eval()
        total_loss = 0
        actual_names_val = []
        predicted_names_val = []
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
                                               max_new_tokens=3)
                predicted_names = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

                actual_names_val += list(method_name)
                predicted_names_val += predicted_names

        avg_val_loss = total_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        chencherry = SmoothingFunction()
        bleu_score_val = corpus_bleu(actual_names_val, predicted_names_val, smoothing_function=chencherry.method1)
        accuracy_val = accuracy_score(actual_names_val, predicted_names_val)

        # avg_rogue_l_val = calculate_rouge_l(predicted_names_val, actual_names_val)
        iou_scores_val = [iou_score(pred, truth) for pred, truth in zip(predicted_names_val, actual_names_val)]
        avg_iou_val = sum(iou_scores_val) / len(iou_scores_val)
        lev_dists_val = [lev.distance(pred, truth) for pred, truth in zip(predicted_names_val, actual_names_val)]
        avg_lev_dist_val = sum(lev_dists_val) / len(lev_dists_val)

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "train_bleu": bleu_score_train,
                   "val_bleu": bleu_score_val, "train_accuracy": accuracy_train, "val_accuracy": accuracy_val,
                   "train_iou": avg_iou_train,
                   "val_iou": avg_iou_val, "train_lev_dist": avg_lev_dist_train, "val_lev_dist": avg_lev_dist_val})

        # wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "train_bleu": bleu_score_train,
        #            "val_bleu": bleu_score_val, "train_accuracy": accuracy_train, "val_accuracy": accuracy_val})


train_and_evaluate(30, tokenizer, model, device, train_loader, eval_loader, optimizer)
