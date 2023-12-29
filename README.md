## Solution Run
If you want to train the model by yourself on your data here is what you should do:

- create venv for Python 3.8+ and activate it
- run `pip install -r requirements.txt`
- run `chmod +x clone.sh`
- run `./clone.sh`
- then run for example `python model/train.py --eval_mode False --num_epochs 10 --batch-size 8` to train model or 
`python model/train.py --eval_mode True` for evaluating pretrained model. You can also evaluate fine-tuned model by running `python model/train.py --eval_mode True --fine_tuned path/to/your/model.pth`
- to see all available flags run `python model/train.py --help`

If you want to read an overview of the completed task you can read `report.md`.
If you want to fully understand the solution please read the code (especially `model/train.py`).

_Note: don't pay attention to errors like `'utf-8' codec can't decode byte ...`_