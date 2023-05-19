import numpy as np
import torch
import deepspeed

### get tokenizer (bert-base-uncased)
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',
                                              sos_token='[SOS]',
                                              eos_token='[EOS]',
                                              pad_token='[PAD]')
print(tokenizer.vocab_size)
print(tokenizer.pad_token)

PAD_TOKEN = '[PAD]'
SOS_TOKEN = '[SOS]'
EOS_TOKEN = '[EOS]'
SOS_TOKEN_ID = tokenizer.convert_tokens_to_ids(SOS_TOKEN)
EOS_TOKEN_ID = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids(PAD_TOKEN)


### get dataset
from tqdm import tqdm
from datasets import load_dataset

train = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')

train_dataset, test_dataset = [], []
for text in tqdm(train['text']):
    if text.count(' ') > 20 and '=' not in text:
        train_dataset.append(
            tokenizer(text, return_tensors='pt', padding='max_length',
                      max_length=256, truncation=True).input_ids.squeeze(0)
        )
torch.save(train_dataset, 'train_dataset.pt')