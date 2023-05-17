import numpy as np
import torch
import deepspeed

###

### parameters from GPT

n_layers = 24
d_model = 2048
n_heads = 32
d_head = 128

### class MPU
class MPU:
    def __init__(self, rank: int, world_size: int, device_ids: list, ):
        self.rank = rank
        self.world_size = world_size
        self.device_ids = device_ids

    def get_model_parallel_rank(self):
        return self.rank

    def get_model_parallel_world_size(self):
        return self.world_size
    
    def get_model_parallel_group(self):
        return self.device_ids

    def get_data_parallel_group(self):
        return self.device_ids

mpu = MPU(rank=0, world_size=4, device_ids=[0, 1, 2, 3])

### hyperparameters

learning_rate = 2e-5

decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=0.1)
model = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(),
                                                    optimizer=optimizer, config='ds_config.json',
                                                    mpu=mpu)

a = torch.randn()