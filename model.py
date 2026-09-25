import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

class LayerNorm(nn.Module):
    def __init__(self, ndim:int, bias:bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        """
        input -> tensor, weight shape, weight, bias, epsilon to tackel zero division error
        1. normalize the input tensor: 
        mean = sum(input)/len(input)
        #--------------------------------
        variance = 0
        for n in input:
        variance += (n - mean)**2
        variance /= len(input)
        #--------------------------------
        normalized = (x - mean) / sqrt(variance + tiny number to tackle divisible by zero error)
        NOTE: mean, variance = integer, tiny = small decimal value, x = Tensor
        #--------------------------------
        2. normalised @ weight + bias
        return the input
        1e-5 = 0.00001
        """
	  
class CausalSelfAttention(nn.Module):
	def __init__(self, config:"GPTConfig"):
		assert config.n_embd % config.n_head == 0
		super().__init__()
		self.n_head : int   = config.n_head
		self.n_embd : int   = config.n_embd
		self.dropout: float = config.dropout
		self.bias   : float = config.bias
		self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=self.bias)
		self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
		'''
		nn.linear
		1. input @ w.T + bias
		2. weight, bias will be created by itself using given shape
		3. nn.Linear(in_features, out_features, bias=True)
		4. weight = torch.randn(out_features, in_features)
		5. bias   = torch.randn(out_features)
		'''
		# remember dropout is always between 0 - 1
		# nn.Dropout(value, inplace=True/False) -> inplace is an optional and this inplace is to create a new tensor or inplace
		self.attn_dropout  = nn.Dropout(self.dropout)
		self.resid_dropout = nn.Dropout(self.dropout)
		#------------------------------------------------------------------------
		'''
		check our PyTorch >= 2.0, then scaled_dot_product_attention will be there
		scaled_dot_product_attention = softmax((Q.K_T)/sqrt(channel)).V
		'''
		
		self.flash = hasattr(F, 'scaled_dot_product_attention')
		if not self.flash: # older version
			print("WARNING: using slow attention. Flash attention requires PyTorch >= 2.0")
			'''
			1. this created tensor will be in the model, but don't train it
			2. this is a templace 
			[1,0,0]
			[1,1,0]
			[1,1,1]
			3. [1] * block_size -> reshape it to -> (1, 1, block_size, block_size)
			4. and to the model buffer
			'''
			self.register_buffer(
					"bias",
					torch.tril(
					torch.ones(config.block_size, config.block_size)
					).view(1, 1, config.block_size, config.block_size)
			)  
		 
	def forward(self, x:Tensor)->Tensor:
		B, T, C = x.shape
		q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
		
		q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
		k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
		v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)

		if self.flash:
			y = F.scaled_dot_product_attention(
				q, k, v, 
				attn_mask=None, 
				dropout_p=self.dropout if self.training else 0, 
				is_causal=True
			)
		else:
			# Q @ K**T -> mask_fill -> softmax -> dropout -> att @ v
			att = q @ k.transpose(1, 2) / math.sqrt(k.size(-1)) # here (1, 2) != (-1, -2) because its a 4D
			att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
			att = F.softmax(att, dim=-1)
			att = self.attn_dropout(att)
			y   = att @ v
		# transpose -> c_proj (2nd Linear) -> resid_proj (2nd Dropout)
		y = y.transpose(-2, -1).contiguous().view(B, T, C)
		y = self.c_proj(y)
		y = self.resid_dropout(y)
		return y
		
class MLP(nn.Module):
	def __init__(self, config:"GPTConfig"):
		super().__init__()
		self.c_fc    = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias) # in_feature = n, out_feature = 4*n
		self.gelu    = nn.GELU()
		self.c_proj  = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias) # in_feature = 4*n, out_feature = n
		self.dropout = nn.Dropout(config.dropout)
		'''
		GELU:
		simplifed aprox: (x/2) * (1 + erf(x/sqrt(2))) # x is a tensor
		tensor version : 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) # not tensor([2])
		erf (normal distribution)-> -1 to 1
		'''
		
	def forward(self, x:Tensor)->Tensor:
		# linear -> GELU -> Linear -> Dropout
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		x = self.dropout(x)
		return x

class Block(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
		self.mlp  = MLP(config)
		
	def forward(self, x:Tensor)->Tensor:
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x

@dataclass
class GPTConfig:
	block_size: int   = 1024
	vocab_size: int   = 50304 # GPT-2 50257, made into multiple of 64
	n_embd    : int   = 768
	n_layer   : int   = 12
	n_head    : int   = 12
	dropout   : float = 0.0
	bias      : bool  = True # True -> Linear/LayerNorm, False -> bit better and faster

class GPT(nn.Module):
	def __init__(self, config):
		assert config.vocab_size is not None and config.block_size is not None
		super().__init__()
		self.config = config

		self.transformer = nn.ModuleDict(dict(
			wte  = nn.Embedding(config.vocab_size, config.n_embd),
			wpe  = nn.Embedding(config.block_size, config.n_embd),
			drop = nn.Dropout(config.dropout),
			h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
			ln_f = LayerNorm(config.n_embd, bias=config.bias)
		))
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		self.transformer.wte.weight = self.lm_head.weight
		self.apply(self._init_weights)

		million = 1e6
		for pn, p in self.named_parameters():
			if pn.endswith('c_proj.weight'):
				nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))
		print(f"Total Parameters: {self.get_num_params()/million:,.2f}M") # 1,234.23
		
		
	def get_num_params(self, non_embedding:bool=True)->float:
		n_param = sum(p.numel() for p in self.parameters())
		if non_embedding:
			n_param -= self.transformer.wpe.weight.numel()
		return n_param
		
	def _init_weights(self, module:Tensor)->None:
		if isinstance(module, nn.Linear):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, idx:Tensor, targets=None):
		b, t = idx.shape
		assert t <= self.config.block_size, f"Cannot forward seqence of length {t}, block size is only {self.config.block_size}" 
		
		device  = idx.device
		pos     = torch.arange(0, t, dtype=torch.long, device=device)
		tok_emb = self.transformer.wte(idx)
		pos_emb = self.transformer.wpe(pos)
		x       = self.transformer.drop(tok_emb + pos_emb)
		
		for block in self.transformer.h:
			x = block(x)
		x = self.transformer.ln_f(x)

		if targets is not None:
			logits = self.lm_head(x)
			loss   = F.cross_entropy(
				logits.view(
					-1, logits.size(-1)
				), 
				targets.view(-1), 
				ignore_index=-1
			)
		else:
			logits = self.lm_head(x[:, [-1], :])
			loss   = None
		return logits, loss

	def crop_block_size(self, block_size:int)->None:
		assert block_size <= self.config.block_size
		
		self.config.block_size      = block_size
		self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
		
		for block in self.transformer.h:
			if hasattr(block.attn, 'bias'):
				block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

	@classmethod
	def from_pretrained(cls, model_type:str, override_args=None)->Tensor:
		assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
		
		override_args = override_args or {}
		assert all(k == 'dropout' for k in override_args)
		#------------------------------------------------
		
		from transformers import GPT2LMHeadModel
		print(f'Loading weights from pretrained GPT: {model_type}')
		config_args = {
			'gpt2'       : dict(n_layer=12, n_head=12, n_embd=768 ),
			'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), 
			'gpt2-large' : dict(n_layer=36, n_head=20, n_embd=1280),
			'gpt2-xl'    : dict(n_layer=48, n_head=25, n_embd=1600),
		}[model_type]

		# As per GPT2 args
		print("Forcing vocab_size=50257, block_size=1024, bias=True")
		config_args['vocab_size'] = 50257
		config_args['block_size'] = 1024
		config_args['bias']       = True

		if 'dropout' in override_args:
			print(f"overriding dropout rate to {override_args['dropout']}")
			config_args['dropout'] = override_args['dropout']

		# create from the scratch, initialized minGPT model
		config  = GPTConfig(**config_args)
		model   = GPT(config)
		sd      = model.state_dict()
		sd_keys = sd.keys()
		sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard bias, bias mask
		#--------------------------------------------------------
		# init Hugging face/transformer model
		# copy it from Hugging face weights to our current model
		model_hf = GPT2LMHeadModel.from_pretrained(model_type)
		sd_hf    = model_hf.state_dict()
		sd_keys_hf = sd_hf.keys()
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]

		# OpenAI checkpoints use a "Conv1D" module but we only want to use a vanilla Linear
		# so that we need to transpose these weights when we import them
		assert len(sd_keys) == len(sd_keys_hf), f"mismatch keys, current_model:{len(sd_keys)} != imported_model:{len(sd_keys_hf)}"
		
		transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] # all are only Linear Function
		for k in sd_keys_hf:
			if any(k.endswith(w) for w in transposed):
				assert sd_hf[k].shape[::-1] == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k].t())
			else:
				# vanilla copy over the other parameters
				assert sd_hf[k].shape == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k])
		return model
		
	def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
		param_dict     = {pn:p for pn, p in self.named_parameters() if p.requires_grad}
		decay_params   = [p for p in param_dict.values() if p.dim() > 1]
		nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
		optim_groups   = [
			 {'params' : decay_params,   'weight_decay' : weight_decay},
			 {'params' : nodecay_params, 'weight_decay' : 0.0}
		]
		
		num_decay_params   = sum(p.numel() for p in decay_params)
		num_nodecay_params = sum(p.numel() for p in nodecay_params)
		
		print(f'# of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters')
		print(f'# of non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters')

		# Create fused AdamW optimizer and use the fused version if its available
		fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
		use_fused = fused_available and device_type == 'cuda'
		extra_args = dict(fused=True) if use_fused else dict()
		
		print(f'using fused AdamW, {use_fused}')
		optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
		return optimizer

	'''
	torch.optim.AdamW, Simplified Explanation

	class AdamW:
		def __init__(self, optim_groups, lr, betas):
			self.groups = optim_groups
			self.lr = lr
			self.beta1, self.beta2 = betas
			self.m, self.v = {}, {}
			self.step = 0
		#----------------------------------------------
		def step(self):
			self.step += 1
			for group in self.groups:
				for w in group["params"]:
					g = w.grad
					if w not in self.m:
						self.m[w] = self.v[w] = 0
						
					# Momentum & Squared_Gradient: 90% older, 10% current 
					self.m[w] = self.m[w] * self.beta1 + (1 - self.beta1) * g
					self.v[w] = self.v[w] * self.beta2 + (1 - self.beta2) * g**2
					
					# Bias correction
					m_hat = self.m[w] / (1 - self.beta1**self.step) 
					v_hat = self.v[w] / (1 - self.beta2**self.step)

					# AdamW
					w -= m_hat * self.lr / (v_hat.sqrt() + 1e-8)
					w -= w     * self.lr * group["weight_decay"]
	'''
	
	def estimate_mfu(self, fwdbwd_per_iter, dt):# need to learn
		# estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
		N   = self.get_num_params()
		cfg = self.config
		L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
		
		'''		
		6 * N:
		2 -> foward pass    
		4 -> baclward pass (2 times of forward_Pass) 
		------------------------------------------
		12*L*H*Q*T:
		QKᵀ  → roughly 2 × T² × H × Q FLOPs
		AV   → roughly 2 × T² × H × Q FLOPs
		4*T²*H*Q:
		forward_Pass = 4*T²*H*Q
		backwark_Pass = 8*T²*H*Q (2 times of forward_Pass)
		------------------------------------------
		embd / H => Q
		H * Q    => embd
		'''
		
		flops_per_token  = 6*N + 12*L*H*Q*T
		flops_per_fwdbwd = flops_per_token * T
		flops_per_iter   = flops_per_fwdbwd * fwdbwd_per_iter
		flops_achieved   = flops_per_iter * (1.0/dt) # per second
		flops_promised   = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
		mfu = flops_achieved / flops_promised
		return mfu

	@torch.no_grad()
	def generate(self, idx:Tensor, max_new_tokens:int, temperature:float=1.0, top_k:int | None=None)->Tensor:
		for _ in range(max_new_tokens):
			idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
			# also called:
			# i = min(idx.size(1), self.config.block_size)
			# idx_cond = idx[:, -i:]
			# forward_pass it to the model, self(idx_cond) mean we passing the idx_cmd to the model
			logits, _ = self(idx_cond) # logits, loss
			logits = logits[:, -1, :] / temperature
			if top_k is not None:
				v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # returns: value, index_position
				logits[logits < v[:, [-1]]] = -float('inf') # sets -inf, if losgits[i] < val[i]
			probs    = F.softmax(logits, dim=-1)
			idx_next = torch.multinomial(probs, num_samples=1)
			idx      = torch.cat((idx, idx_next), dim=1) 
			
			'''
			i guess dim = 2
				idx      |   idx_next
			--------------------------
			[[1, 2, 3], | [[9, 10],
			[5, 6, 7]]  |  [11, 12]]
			---------------------------
			result:
			[[1, 2, 3, 9, 10],
		     [5, 6, 7, 11, 12]]
			'''
		return idx
			
