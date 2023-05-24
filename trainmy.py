import numpy as np
import torch
import os

# from transformers import GPT2TokenizerFast
from datasets import load_dataset

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

def tokenize_sentence(sentence, max_length=None):
    if max_length:
        return tokenizer(sentence, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True).input_ids
    else:
        return tokenizer(sentence, return_tensors='pt').input_ids

class PositionwiseFeedForwardLayer(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.linear1 = torch.nn.Linear(d_model, 4 * d_model)
        self.linear2 = torch.nn.Linear(4 * d_model, d_model)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(self.linear2(x))

        # x shape == output shape
        return x

class Head(torch.nn.Module):
    def __init__(self, d_model: int, d_head: int, dropout: float):
        super().__init__()

        assert d_model % d_head == 0
        d_tensor = d_model // d_head
        self.d_tensor = d_tensor

        self.key = torch.nn.Linear(d_model, d_head)
        self.query = torch.nn.Linear(d_model, d_head)
        self.value = torch.nn.Linear(d_model, d_head)

        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, q, k, v):

        # q, k, v = (batch_size, seq_len, d_model)

        q, k = self.query(k), self.key(q)


        # q, k = (batch_size, seq_len, d_tensor)
        # kT = (batch_size, d_tensor, seq_len)


        wei = q @ k.transpose(-2, -1) * (self.d_tensor ** (-0.5)) # q*kT/sqrt(d_k) from paper "Attention is All You Need"



        # wei = (batch_size, seq_len, seq_len)
        
        wei = torch.nn.functional.softmax(wei, dim=-1)
        v = self.value(v)



        # wei = (batch_size, seq_len, seq_len)
        # v = (batch_size, seq_len, d_tensor)

        out = wei @ v



        # out = (batch_size, seq_len, d_tensor): d_tensor * n_heads = d_model

        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, dropout: float, num_gpus: int):
        super().__init__()

        assert d_model % d_head == 0
        assert n_heads % num_gpus == 0
        d_tensor = d_model // d_head
        self.d_tensor = d_tensor

        self.heads = torch.nn.ModuleList([
            Head(d_model=d_model, d_head=d_head, dropout=dropout) for _ in range(n_heads)
        ])
        self.linear = torch.nn.Linear(n_heads * d_tensor, d_model) # n_heads * d_tensor == d_model
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, q, k, v):

        out = torch.cat([
            head(q, k, v) for head in self.heads
        ], dim=-1)
        


        return out

class LayerNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps=1e-12):
        super().__init__()

        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x. var(-1, unbiased=False, keepdim=True)

        out = (x - mean) * ((var + self.eps) ** (-0.5))
        out = self.gamma * out + self.beta

        return out

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, dropout: float, device_num: int):
        super().__init__()

        self.device_num = device_num
        self.attention_layernorm = LayerNorm(d_model)
        self.feedforward_layernorm = LayerNorm(d_model)

        self.self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_head=d_head, dropout=dropout, num_gpus=4)
        self.positionwise_feedforward = PositionwiseFeedForwardLayer(d_model=d_model, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, trg):

        # trg = (batch_size, seq_len, d_model)
        # trg_mask = (batch_size, seq_len)

        trg = trg.to(device[self.device_num])
    
        # self attention with dropout
        _trg = self.dropout(self.self_attention(trg, trg, trg))

        # _trg = (batch_size, seq_len, d_model) == trg
        # add & norm with residual connection

        trg = self.attention_layernorm(trg + _trg)


        # trg = (batch_size, seq_len, d_model)
        # positionwise feedforward layer
        _trg = self.dropout(self.positionwise_feedforward(trg))
        trg = self.feedforward_layernorm(_trg + trg)

        # trg = (batch_size, seq_len, d_model)
        return trg

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_head: int, max_length: int, dropout: float, num_gpus: int):
        super().__init__()

        # positional encoding
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model).to(device[0])
        self.position_embedding = torch.nn.Embedding(max_length, d_model).to(device[0])

        self.n_layers = n_layers
        self.per_gpu = n_layers // num_gpus # 3
        print(f"{self.per_gpu} decoder layers per gpu, with {num_gpus} gpus")
        self.layers = torch.nn.ModuleList([
            *[DecoderLayer(d_model=d_model, n_heads=n_heads, d_head=d_head, dropout=dropout, device_num=0).to(device[0]) for _ in range(self.per_gpu)],
            *[DecoderLayer(d_model=d_model, n_heads=n_heads, d_head=d_head, dropout=dropout, device_num=1).to(device[1]) for _ in range(self.per_gpu)],
            *[DecoderLayer(d_model=d_model, n_heads=n_heads, d_head=d_head, dropout=dropout, device_num=2).to(device[2]) for _ in range(self.per_gpu)],
            *[DecoderLayer(d_model=d_model, n_heads=n_heads, d_head=d_head, dropout=dropout, device_num=3).to(device[3]) for _ in range(self.per_gpu)],
        ])

        self.fc_out = torch.nn.Linear(d_model, vocab_size).to(device[3])
        self.dropout = torch.nn.Dropout(dropout).to(device[0])
    
    def forward(self, trg):
        
        # trg = (batch_size, seq_len)
        # trg_mask = (batch_size, seq_len)

        batch_size, seq_len = trg.shape

        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(device[0])
        trg = self.dropout((self.token_embedding(trg) + self.position_embedding(pos)))

        # trg = (batch_size, seq_len, d_model)

        # Decoder layers


        for layer in self.layers:

            trg = layer(trg)

        
        # trg = (batch_size, seq_len, d_model)

        output = self.fc_out(trg)

        # output = (batch_size, seq_len, vocab_size)

        return output

class GPTModel(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_head: int, max_length: int, dropout: float, tokenizer, num_gpus: int=4):
        super().__init__()

        self.tokenizer = tokenizer
        self.decoder = Decoder(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_head=d_head, max_length=max_length, dropout=dropout, num_gpus=num_gpus)
    

    def forward(self, sentence: str):
        '''
        This is used for the inference for the next word prediction for each batch, each word.
        '''
            
        # trg = (batch_size, seq_len)
        # trg_mask = (batch_size, seq_len)

        trg = self.tokenizer(sentence, return_tensors='pt').input_ids.to(device[0])
        output = self.decoder(trg)

        # output = (batch_size, seq_len, vocab_size)

        return output
    
    def next_word_prediction(self, sentence):
        '''
        This is used for the inference for the next word prediction for each batch using my decoder
        '''
        with torch.no_grad():
            # trg = (batch_size, seq_len)
            # trg_mask = (batch_size, seq_len)
            trg = self.tokenizer(sentence, return_tensors='pt').input_ids.to(device[0])
            trg = trg.to(device[0])
            out = torch.argmax(self.decoder(trg)[:, -1, :], dim=-1)

            output = []
            for item in out:
                output.append(self.tokenizer.decode(item))

            # output = (batch_size, vocab_size) (next word prediction)

            return output

    def generate(self, sentence, max_length=20):
        '''
        This is used for making the prediction over and over again until the end token is predicted.
        (or reached max_length)
        '''
        with torch.no_grad():
            trg = self.tokenizer(sentence, return_tensors='pt').input_ids.to(device[0])

            for _ in range(max_length):
                trg = trg.to(device[0])
                out = torch.argmax(self.decoder(trg)[:, -1, :], dim=-1).to(device[0])
                trg = torch.cat((trg, out.unsqueeze(1)), dim=1)
                if out == EOS_TOKEN_ID:
                    break
            
            return self.tokenizer.decode(trg[0])

    def train(self, full_sentence: list, loss_fn, optimizer, max_length=20):
        '''
        With given sentence, it will generate the next word prediction and backpropagate the loss.
        '''
        
        longest = 0
        for i in range(len(full_sentence)):
            full_sentence[i] = np.concatenate(
                ([SOS_TOKEN_ID], tokenizer(full_sentence[i]).input_ids, [EOS_TOKEN_ID]), axis=-1,
            )
            longest = full_sentence[i].shape[0] if full_sentence[i].shape[0] > longest else longest
            full_sentence[i] = np.expand_dims(full_sentence[i], axis=0)

        batched = torch.tensor(
            [token_full[0][i:i+max_length] for i in range(0, len(token_full[0])-max_length)]
        ).to(device[0])
        answer = torch.tensor(
            [token_full[0][i:i+max_length] for i in range(1, len(token_full[0])-max_length+1)]
        ).to(device[-1])

        self.decoder.zero_grad()
        optimizer.zero_grad()

        output = self.decoder(batched)
        loss = loss_fn(output.view(-1, output.shape[-1]), answer.view(-1))

        loss.backward()
        optimizer.step()

        ret = loss.item()

        del token_full, batched, answer, output, loss
        torch.cuda.empty_cache()

        return ret

# hyperparameters

# model hyperparameters (from GPT3 XL)
n_layers = 24
d_model = 2048
n_heads = 32
d_tensor = d_model // n_heads # => 64
d_head = 64
max_length = 128

vocab_size = tokenizer.vocab_size + 3 # +3 for <sos>, <eos>, <pad>
dropout = 0.1
batch_size = 64
learning_rate = 2e-5
num_epochs = 5

# print(f"vocab_size = {vocab_size}")
# print(f"d_tensor = {d_tensor}")
# print(f"d_model = {d_model}")
# print(f"d_tensor * n_heads = {d_tensor * n_heads}")

model = GPTModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                 n_heads=n_heads, d_head=d_head, max_length=max_length,
                 dropout=dropout, tokenizer=tokenizer)

print(f"parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):_}")
