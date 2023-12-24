import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torchmetrics import BLEUScore
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import accuracy_score

from dataset.dataset import MethodNameDataset

checkpoint = "Salesforce/codet5p-770m"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

dataset = MethodNameDataset('../intellij-community')
eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)
print(f'Number of samples: {len(dataset)}')

actual_names = []
predicted_names = []


def predict_method_name(body):
    inputs = tokenizer.encode(body, return_tensors='pt').to(device)
    if inputs[0].shape[0] > 512:
        inputs = inputs[:, :512]
    outputs = model.generate(inputs, max_length=10)
    predicted_name = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_name


for method_body, actual_name in tqdm(eval_loader):
    predicted_name = predict_method_name(method_body[0])
    actual_names.append([actual_name])
    predicted_names.append(predicted_name)

bleu_score = corpus_bleu(actual_names, predicted_names)

flat_actual_names = [name for sublist in actual_names for name in sublist]
flat_predicted_names = predicted_names

accuracy = accuracy_score(flat_actual_names, flat_predicted_names)

print(f"BLEU Score: {bleu_score}")
print(f"Accuracy: {accuracy}")
