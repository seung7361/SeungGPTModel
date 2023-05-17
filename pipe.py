import numpy as np
import torch
import deepspeed
from torch.distributed.pipeline.sync import Pipe

### initializate

torch.distributed.init_process_group(backend='nccl', init_method='env://')
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=4)
device = [
    torch.device('cuda:0'),
    torch.device('cuda:1'),
    torch.device('cuda:2'),
    torch.device('cuda:3'),
]

### parameters from GPT

n_layers = 24
d_model = 1024
n_heads = 16
d_head = 64

### hyperparameters

learning_rate = 2e-5

decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=0.1)
model = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers)
print('model has {} parameters'.format(sum(p.numel() for p in model.parameters())))

pipe = Pipe(model, chunks=4)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

a = torch.randint(low=0, high=52600, size=(32, 1024), dtype=torch.long).to(device[0])
b = torch.randint(low=0, high=52600, size=(32, 1), dtype=torch.long).to(device[0])
c = pipe(a, b)

print(c)