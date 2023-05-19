import numpy as np
import torch
import deepspeed

### initializate

deepspeed.init_distributed()

### parameters from GPT

n_layers = 24
d_model = 1024
n_heads = 16
d_head = 64

### hyperparameters

vocab_size = 52600

### hyperparameters

learning_rate = 2e-5

decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=0.1)
model = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers)
print('model has {:_} parameters'.format(sum(p.numel() for p in model.parameters())))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(),
                                                    optimizer=optimizer, config='ds_config.json')

a = torch.randn(32, 1024).cuda()
b = torch.randn(32, 1024).cuda()
c = model_engine(a, b)

ans = torch.randn(32, 1024).cuda()

print(c)
print(c.shape)

loss = loss_fn(c, ans)
loss.backward()
optimizer.step()

# sleep
import time
time.sleep(10)