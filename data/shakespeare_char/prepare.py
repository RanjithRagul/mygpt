import os
import pickle
import requests
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
  data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
  with open(inpput_file_path, 'w') as f:
    f.write(request.get(data_url)).text)

with open(input_file_path, 'r') as f:
  data = f.read()
print(f'length of dataset in char: {len(data):,}')

chars = sorted(list(set(data)))
vocab_size = len(chars)
print('all the unique character:', ''.join(chars))
print(f'vocab size: {vocab_size:,}')

# create a mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

def encode(s: str) -> list[int]:
  return [stoi[c] for c in s]

def decode(l: list[int]) -> str:
  return ''.join([itos[i] for i in l])

N = len(data)
train_data = data[:int(N * 0.9)] # 90%
val_data   = data[int(N * 0.9):] # 10%

train_ids = encode(train_data)
val_ids   = encode(val_data)
print(f'train has {len(train_ids):,} tokens')
print(f'val has {len(val_ids):,} tokens')

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids   = np.array(val_ids,   dtype=np.uint16)

meta = {
  'vocab_size': vocab_size, # int
  'itos'      : itos,       # dict
  'stoi'      : stoi,       # dict
}
