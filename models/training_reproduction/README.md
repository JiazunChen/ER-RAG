Meta KDD Cup '24 CRAG: team \textbf{db3} solution: training part reproduction

This repository is the training reproduction directory for the team \textbf{db3} solution. 

We will provide the training data and training codes for the checkpoint our solution uses.

1. the model for answering web context queries: "models/pretrain_models/llama3-52-peft/checkpoint-480":
    the training file: train_516.txt
    the training code: peft-516.ipynb
    max_steps=600
    checkpoint_steps=480
2. the model for generating APIs (excluding sports): "models/pretrain_models/llama3-52-peft/checkpoint-500":
    the training file: train_523.txt
    the training code: peft-523-api.ipynb
    max_steps=500
    checkpoint_steps=500
3. the model for answering API extraction context queries (excluding sports): "models/pretrain_models/llama3-52-peft/checkpoint-580":
    the training file: train_524.txt
    the training code: peft-524-api.ipynb
    max_steps=580
    checkpoint_steps=580
4. the model for generating APIs: "models/pretrain_models/train_618api_up/checkpoint-580":
    the training file: train_618api.txt
    the training code: peft-617_api.py
    max_steps=580
    checkpoint_steps=580
5. the model for answering API extraction context queries: "models/pretrain_models/tran_619_apioutput/checkpoint-310":
    the training file: train_619_apioutput.txt
    the training code: peft-619_apioutput.ipynb
    max_steps=310
    checkpoint_steps=310

p.s.: the steps above are all set under, device_number=2, per_device_train_batch_size = 1, gradient_accumulation_steps = 4
