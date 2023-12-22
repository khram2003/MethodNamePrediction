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
    bleu = BLEUScore()

    with torch.no_grad():
        for inputs, targets in tqdm(eval_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model.generate(inputs, max_length=10)
            outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
            bleu.update(outputs, targets)

    return bleu.compute()


if __name__ == '__main__':
    bleu = evaluate_pretrained_model()
    print(f'BLEU: {bleu}')
