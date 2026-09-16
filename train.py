import os
import math
import time
import pickle
from contextlib import nullcontext

import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
#------------------------------ dir --------------------------------------------
out_dir = 'out'
dataset = 'openwebtext'
#------------------------------ config-1 ---------------------------------------
batch_size = 12
block_size = 1024
n_layer = 12
n_head  = 12
n_embd  = 768
dropout = 0.0
bias    = False
gradient_accumulation_steps = 5 * 8
#------------------------------ config-2 ---------------------------------------
init_from     = 'scratch'
eval_interval = 2000
log_interval  = 1
eval_iters    = 200
eval_only     = False
always_save_checkpoint = True
#------------------------------ wandb logging ----------------------------------
wandb_log      = False
wandb_project  = 'owt'
wandb_run_name = 'gpt2'
#------------------------------ adamW optimizer --------------------------------
learning_rate = 6e-4 # 0.6000
max_iters = 600000 
weight_decay = 1e-1 # 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
#--------------------------------------------------------------------------
device = 'cuda'
gradient_Accumulation = 5 * 8
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
#--------------------------------------------------------------------------
backend = 'nccl' # gloo, etc
ddp     = int(os.environ.get('RANK', -1)) != -1

if ddp: # GPU
  init_process_groups(backend=backend)
  ddp_rank       = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  assert gradient_Accumulation % ddp_world_size == 0

  master_process = ddp_rank ==  0
  seed_offset    = ddp_rank
  device         = f'cuda:{dpp_local_rank}'
  torch.cuda.set_device(device)
  gradient_accumulation_stes //= ddp_world_size
else: # CPU
  master_process = True
  seed_offset    = 0
  ddp_world_size = 1

token_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f'toeks per iteration will be: {token_per_iter:,}')

if master_process:
  os.makedirs(out_dir, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype     = {
                'bfloat16' : torch.bfloat16,
                'float16'  : torch.float16,
                'float32'  : torch.float32,
              }[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
def get_batch(split):
  if split == 'train':
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
  else:
    data = np.memmap(os.path.join(data_dir, 'val.bin'  ), dtype=np.uint16, mode='r')
    
  ix = torch.ranint(len(data) - block_size, (batch_size,))
  x  = torch.stack([torch.from_numpy((data[i  :i+block_size ]).astype(np.int64))  for i in ix])
  y  = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in id])

  if device_type == 'cuda':
    x = x.pin_memory().to(device, non_blocking=True)
    y = y.pin_memory().to(device, non_blocking=True)
  else:
    x = x.to(device)
    y = y.to(device)
  return x, y

# init these up here, can override if init_from='resume' (i.e, from a checkpoint)
iter_num = 0
bext_val_loss = 1e9 # 3K, 6M, 9B, 12T

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None

if os.path.exists(meta_path):
  with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
  meta_vocab_size = meta['vocab_size']
  print(f'found vocab_size: {meta_vocab_size}, inside: {meta_path}')

model_args   = dict(
                    block_size = block_size,
                    vocab_size = None,
                    n_embd     = n_embd,
                    n_layer    = n_layer,
                    n_head     = n_head,
                    dropout    = dropout,
                    bias       = bias
                  )

if init_from == 'scratch':
  print(f'Initializing a new model from scratch')
  if meta_vocab_size is None:
    print('Defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)')
    
  model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
  gptconfig = GPTConfig(**model_args)
  model = GPT(gptconfig)
  
elif init_from == 'resume':
  print(f'Resuming training from {out_dir}')
  ckpt_path  = os.path.join(out_dir, 'ckpt.pt')
  checkpoint = torch.load(ckpt_path, map_location = device)
  checkpoint_model_args = checkpoint['model_args']
  
  for k in ['block_size', 'vocab_size', 'n_embd', 'n_layer', 'n_head', 'bias']:
    model_args[k] = checkpoint_model_args[k]

  gptconfig = GPTConfig(**model_args)
  model = GPT(gptconfig)
  
  state_dict = checkpoint['model']
  unwanter_prefix = '_orgin_mod.'
  N_unwanter_prefix = len(unwanted_prefix)
  
  for k in state_dict.keys():
    if k.startswith(unwanted_prefix):
      state_dict[k[N_unwanted_prefix:]] = state_dict.pop(k)
      
  model.load_state_dict(state_dict)
  iter_num = checkpoint['iter_num']
  best_val_loss = checkpoint['best_val_loss']
  
elif init_from.startwith('gpt2'):
  print('Initializing from OpenAI GPT-2 weights: {init_from}')
  override_args = dict(dropout=dropout)
  model = GPT.from_pretrained(init_from, override_args)
  for k in ['block_size', 'vocab_size', 'n_embd', 'n_layer', 'n_head', 'bias']:
    model_args[k] = getattr(model.config, k)

# cropdown the model block size if desired, using model surgery
if block_size < model.config.block_size:
  mode.crop_block_size(block_size)
  model_args['block_size'] = block_size
  
model.to(device)
'''
Usage of GradScaler:
for float32: use the precise data
but float16, 0.00000001 set to be 0. we will lost the accuracy
0. small gard -> multiple by large # -> gard becomes representable in FP16 -> optimizer update -> divide by same #
1. loss = 0.0001
2. scale = 63336
3. scaled_loss = loss * scale
4. => 6.5536
5. scaler.scale(loss).backward()
6. check Inf/NaN
7. ÷ same scale
8. optimizer.step()
9. update scale
'''
scaler = torch.cuda.amp.GradScaler(enabled=dtype=='float16')
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
  optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# complie the model
if compile:
  print('Compiling the model...')
  unoptimized_model = model
  model = torch.compile(model) # torch >= 2.0

if ddp:
  model = DDP(model, devive_ids = [ddp_local_rank])

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      with cts:
        logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split]  = losses.mean()
  model.train()
  return out

def get_lr(it:float) -> float:
  if it < warmup_iters:
    return learning_rate * (it + 1) / (warmup_iters + 1)
  if lr_decayiters < it:
    return min_lr
  decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
  coeff       = 0.5 * (1 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (learning_rate - min_lr)
  
if wandb_log and master_process:
  import wandb
  wandb.init(project=wandb_project, name=wandb_run_name, config=config)
  
X, Y = get_batch('train')
t0   = time.time()
running_mfu    = -1.0
local_iter_num = 0
raw_model = model.module if ddp else model

while True:
  lr = get_lr(iter_num) if decay_lr else learning_rate
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  if iter_num % eval_interval and master_process:
    losses = estimate_loss()
    print(f'step: {iter_num}, train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}')
  
    if wandb_log:
      wandb.log({
        'lr'        : lr,
        'iter'      : iter_name,
        'mfu'       : running_mfu * 100,
        'train/loss': losses['train'],
        'val/loss'  : losses['val'],
      })

    if losses['val'] < best_val_loss or always_save_checkpoint:
      best_val_loss = losses['val']
      if 0 < iter_num:
        checkpoint = {
          'iter_num'     : iter_num,
          'best_val_loss': best_val_loss,
          'model_args'   : model_args,
          'config'       : config,
          'model'        : raw_model.state_dict(),
          'optimizer'    : optimizer.state_dict(),
        }
        print(f'saving checkpoint to {out_dir}')
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

  if iter_num == 0 and eval_only:
    break

  for micro_step in range(gradient_accumulation_steps):
    if ddp:
      model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps-1) # last  iteration
    with ctx:
      logits, loss = model(X, Y)
      loss = loss / gradient_accumulation_steps
      
    X, Y = get_batch('train')
    scaler.scale(loss).backward()

  if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
  scaler.step(optimizer)
  optimizer.zero_grad(set_to_none = True)

  # timing and logging
  t1 = time.time()
  dt = t1 - t0
  t0 = t1
  if iter_num % log_interval == 0 and master_processs:
    lossf = loss.item() * gradient_accumulation_steps
    if 4 < local_iter_num:
      mfu         = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
      running_mfu = mfu if running_mfu == -1 else (0.9 * running_mfu) + (0.1 * mfu) # 90% -> prev + 10% current
  print(f'iter: {iter_num}, loss: {lossf:.4f}, time: {dt * 1000:.2f}ms, mfu: {running_mfu * 100:.2f}%')
  
  iter_num += 1
  local_iter_num += 1
  
  if max_iters < iter_num:
    break

if ddp:
  destory_process_group()
