import numpy as np
import torch
import deepspeed

### get tokenizer (bert-base-uncased)
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',
                                              sos_token='[SOS]',
                                              eos_token='[EOS]',
                                              pad_token='[PAD]')

PAD_TOKEN = '[PAD]'
SOS_TOKEN = '[SOS]'
EOS_TOKEN = '[EOS]'
SOS_TOKEN_ID = tokenizer.convert_tokens_to_ids(SOS_TOKEN)
EOS_TOKEN_ID = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids(PAD_TOKEN)


### get dataset
from tqdm import tqdm
from datasets import load_dataset

# train = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')

# train_dataset, test_dataset = [], []
# for text in tqdm(train['text']):
#     if text.count(' ') > 20 and '=' not in text:
#         train_dataset.append(
#             tokenizer(text, return_tensors='pt', padding='max_length',
#                       max_length=256, truncation=True).input_ids.squeeze(0)
#         )
train_dataset = torch.load('train_dataset.pt')
### initializate

deepspeed.init_distributed()

### parameters from GPT (Medium)

# n_layers = 24
# d_model = 1024
# n_heads = 16
# d_head = 64

### parameters from GPT (XL)

n_layers = 24
d_model = 2048
n_heads = 32
d_head = 128

### parameters from GPT (7B)

# n_layers = 32
# d_model = 4096
# n_heads = 32
# d_head = 128

### hyperparameters

vocab_size = tokenizer.vocab_size + 3

### hyperparameters

learning_rate = 2e-5

decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=0.1)
model = torch.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layers)
print('model has {:_} parameters'.format(sum(p.numel() for p in model.parameters())))

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

model_engine, optimizer, train_loader, _ = deepspeed.initialize(model=model,
                                                     model_parameters=model.parameters(),
                                                    optimizer=optimizer,
                                                    training_data=train_dataset,
                                                    config='ds_config.json')

num_epochs = 5
emb = torch.nn.Embedding(vocab_size, d_model)

for epoch in range(num_epochs):
    for step, data in tqdm(enumerate(train_loader)):

        put, ans = emb(data[:, :-1]).cuda(), emb(data[:, 1:]).cuda()

        loss = loss_fn(model_engine(put, ans), ans)

        model_engine.backward(loss)
        model_engine.step()
        
        print(loss.item())



# sleep
import time
time.sleep(10)
