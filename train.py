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

# ------------------------------ dir --------------------------------------------
out_dir = 'out'
dataset = 'openwebtext'

# ------------------------------ config-1 ---------------------------------------
# B³N³DG
batch_size = 12
block_size = 1024
bias = False
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
gradient_accumulation_steps = 5 * 8

# ------------------------------ config-2 ---------------------------------------
# AILEEE
always_save_checkpoint = True
init_from = 'scratch'
log_interval = 1
eval_interval = 2000
eval_iters = 200
eval_only = False

# ------------------------------ wandb logging ----------------------------------
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'

# ------------------------------ adamW optimizer --------------------------------
# (BMW)² LC DG, bmw local dog 
compile = True
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# ------------------------------ config loader ----------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}

# ------------------------------ ddp & device setup -----------------------------
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
backend = 'nccl'
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    assert gradient_accumulation_steps % ddp_world_size == 0

    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f'tokens per iteration will be: {tokens_per_iter:,}')
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337 + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ------------------------------ data loader ------------------------------------
data_dir = os.path.join('data', dataset)

def get_batch(state):
    if state == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

# init counters
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    if master_process:
        print(f'found vocab_size: {meta_vocab_size}, inside: {meta_path}')

model_args = dict(
    block_size=block_size,
    vocab_size=None,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    dropout=dropout,
    bias=bias,
)

if init_from == 'scratch':
    if master_process:
        print('Initializing a new model from scratch')
    if meta_vocab_size is None and master_process:
        print('Defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)')

    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconfig = GPTConfig(**model_args)
    model = GPT(gptconfig)
elif init_from == 'resume':
    if master_process:
        print(f'Resuming training from {out_dir}')
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    for k in ['block_size', 'vocab_size', 'n_embd', 'n_layer', 'n_head', 'bias']:
        model_args[k] = checkpoint_model_args[k]

    gptconfig = GPTConfig(**model_args)
    model = GPT(gptconfig)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    if master_process:
        print(f'Initializing from OpenAI GPT-2 weights: {init_from}')
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['block_size', 'vocab_size', 'n_embd', 'n_layer', 'n_head', 'bias']:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)

model.to(device)

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# compile the model
if compile:
    if master_process:
        print('Compiling the model...')
    model = torch.compile(model)

@torch.no_grad()
def estimate_loss():
    out = {}
    raw_model.eval()
    for state in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(state)
            with ctx:
                logits, loss = raw_model(X, Y)
            losses[k] = loss.item()
        out[state] = losses.mean()
    raw_model.train()
    return out

def get_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

t0 = time.time()
running_mfu = -1.0
local_iter_num = 0

# ------------------------------ main training loop ----------------------------
while True:
    if iter_num > max_iters:
        break

    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluation and checkpointing
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step: {iter_num}, train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}")

        if wandb_log:
            wandb.log({
                'lr': lr,
                'iter': iter_num,
                'mfu': running_mfu * 100,
                'train/loss': losses['train'],
                'val/loss': losses['val'],
            })

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'config': config,
                    'optimizer': optimizer.state_dict(),
                    'model': raw_model.state_dict(),
                    'model_args': model_args,
                    'best_val_loss': best_val_loss,
                    'iter_num': iter_num,
                }
                print(f'saving checkpoint to {out_dir}')
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if eval_only:
        break

    # forward and backward passes with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        X, Y = get_batch('train')
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    if device_type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num > 4:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1 else (0.9 * running_mfu) + (0.1 * mfu)
        print(f'iter: {iter_num}, loss: {lossf:.4f}, time: {dt * 1000:.2f}ms, mfu: {running_mfu * 100:.2f}%')

    iter_num += 1
    local_iter_num += 1

if ddp:
    destroy_process_group()
