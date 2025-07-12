# JunQi
a self-trained NN for JunQi based on DeepNash

|pth|learning rate|history depth|total timestep|
|-|-|-|-|
|PPO_JunQi_0_2_0|0.004|50|2e5|
|PPO_JunQi_0_4_0|0.004|50|1e6|

## setup
firstly, install the base enviroment:
```
pip3 install -r requirements.txt
```
if you want to train or run it on GPU instead of CPU, add:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
to train a new model, modify the file and try:
```
python src/train.py
```
to test (output the game state and info into logs/x.x/) or battle with each other upon existed models, modify these files and try:
```
python src/test.py
python src/battle.py
```

## description
- data/: existed models (PyTorch weight files)
- docs/: DeepNash theoretical paper
- logs/: training logs & game board states (two steps each image)
- resources/: resource files for src/JunQi/wboard.py
- src/: JunQi/ for JunQi enviroment, Nash/ for nash NN model & model starters
