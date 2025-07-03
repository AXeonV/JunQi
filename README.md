# JunQi
a self-trained NN for JunQi based on DeepNash

## setup
firstly, install the base enviroment:
```
pip3 install -r requirements.txt
```
if you want to train or run it on GPU instead of CPU, add:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
to train a new model, modify src/train.py and try:
```
python src/train.py
```
to test (output the game state and info into logs/x.x/) or battle with each other upon existed models, modify files and try:
```
python src/test.py
python src/battle.py
```

## description
- data/: existed models (PyTorch weight files)
- docs/: DeepNash theoretical paper
- logs/: training logs & game board states (two steps each image)
- resources/: resource files for src/JunQi/wboard.py
- src:/ JunQi/ for JunQi enviroment, Nash for nash NN model & model starters
