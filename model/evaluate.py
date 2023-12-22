import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torchmetrics import BLEUScore

from dataset.dataset import MethodNameDataset

checkpoint = "Salesforce/codet5p-770m"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

dataset = MethodNameDataset('../intellij-community')
eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)
print(f'Number of samples: {len(dataset)}')


def evaluate_pretrained_model():
    model.eval()
    bleu = BLEUScore()

    with torch.no_grad():
        for bodies, names in tqdm(eval_loader):
            inputs = tokenizer.encode(bodies[0], return_tensors='pt').to(device)
            if inputs[0].shape[0] > 512:
                continue
            outputs = model.generate(inputs, max_length=10)
            predicted_name = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(predicted_name)
            bleu.update(predicted_name, names[0])

    return bleu.compute()


if __name__ == '__main__':
    bleu = evaluate_pretrained_model()
    print(f'BLEU: {bleu}')
