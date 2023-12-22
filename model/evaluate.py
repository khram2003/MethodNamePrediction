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

dataset = MethodNameDataset('../../intellij-community')
eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)


def evaluate_pretrained_model():
    model.eval()
    # accuracy = Accuracy()
    bleu = BLEUScore()

    with torch.no_grad():
        for idx, (method_bodies, method_names) in enumerate(tqdm(eval_loader)):
            inputs = tokenizer.encode(method_bodies, return_tensors="pt", padding=True, truncation=True).to(device)

            outputs = model.generate(**inputs)
            predictions = tokenizer.decode(outputs, skip_special_tokens=True)

            bleu.update(preds=predictions, target=method_names)

    return bleu.compute()


if __name__ == '__main__':
    bleu = evaluate_pretrained_model()
    print(f'BLEU: {bleu}')
