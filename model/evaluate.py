import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torchmetrics import BLEUScore

from dataset.dataset import MethodNameDataset

checkpoint = "Salesforce/codet5-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

dataset = MethodNameDataset('../intellij-community')
eval_loader = DataLoader(dataset, batch_size=32, shuffle=False)
print(f'Number of samples: {len(dataset)}')

def evaluate_pretrained_model():
    model.eval()
    # accuracy = Accuracy()
    bleu = BLEUScore()

    with torch.no_grad():
        for method_bodies, method_names in tqdm(eval_loader):
            method_bodies = method_bodies.to(device)
            method_names = method_names.to(device)

            inputs = tokenizer(method_bodies, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model.generate(method_bodies, max_length=10)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            bleu.update(outputs, method_names)


if __name__ == '__main__':
    bleu = evaluate_pretrained_model()
    print(f'BLEU: {bleu}')
