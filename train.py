import numpy as np
import torch
import deepspeed
import mpu

### parameters from GPT

n_layers = 24
d_model = 2048
n_heads = 32
d_head = 128

### hyperparameters

learning_rate = 2e-5

decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=0.1)
model = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

deepspeed.init_distributed()
print('deepspeed enabled.')

model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), mpu=mpu, optimizer=optimizer)