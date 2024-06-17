# cordvox
Experiments of neural vocoder

# Installation
```sh
pip3 install -r requirements.txt
```

# Train
1. preprocess dataset
```sh
python3 preprocess.py /jvs_ver1
```

2. run training without discriminator (optional, for fast training)
```sh
python3 train.py -nod True
```

3. runt training with discrimonator
```
python3 train.py 
```

# Inference
```sh
python3 infer_webui.py
```