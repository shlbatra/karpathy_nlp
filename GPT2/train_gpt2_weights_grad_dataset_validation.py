# Run across multiple GPUs - divide batch across 8 devices and then combine them together as an average
# Dataset used is https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
# Used LLM to filter data which is very high education content only

import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
# from hellaswag import render_example, iterate_examples

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really bias , more of mash, but following openai/hf naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x): # 1024 tokens  -> each token has query, key and value,
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) - B & nh as batches and run in parallel
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all queries and keys)

        # Replace flash attention - kernel fusion torch compile cannot find as algo rewrite needed - 7.6 faster - shared and high value memory - att matrix not materialized to hbm - evaluate softmax on online manner
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #  query * key - attention amount ; 
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # token attend previously
        # att = F.softmax(att, dim=-1) # normalize attention
        # y = att @ v # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs); weighted sum of values of interesting token
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd) # 2 linear layers with gelu non linearity
        self.gelu    = nn.GELU(approximate='tanh') # fix dead neuron problem - activation fall less than 0 get 0 gradient, gelu always contribute local gradient
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x) # smoother relu
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # clean residual path way, pre normalization then attention - tokens communicate - aggregate or reduce opn
        x = x + self.mlp(self.ln_2(x)) # pre normalization then mlp - every single token individually
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50304 #this number can be better divided by 2; 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    # new tokens not used -> never index in those rows - memory not used - network learn prob for these tokens to 0. calculations done chunk in 32/64 - no boundary kernels reqd by padding inputs
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config): # uses random weights
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme between embedding table at input and output of final linear layer
        self.transformer.wte.weight = self.lm_head.weight # single tensor used twice 768*50257 params saved. 33% of params saved in the model.

        self.apply(self._init_weights) # apply method from nn module on all sub modules
 
    def _init_weights(self, module): # matching initialization same as gpt paper
        std = 0.02 # 1/ sqrt(768) = 0.03
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * self.config.n_layer) ** -0.5 # for accumulation of residual path; 1 / sqrt(nnn ); 2 * for attention and mlp
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, idx, targets=None): # idx token
        # idx is of shape (B, T)
        B, T = idx.size() # batch size by time - B sequences of token size T
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd) -> broadcast to (B, T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # pos_emb same for each row so broadcasting happening here
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size) -> (B, T+1) - softmax to get probabilitues
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # B*T, Vocab size
        return logits, loss

    @classmethod # return GPT object
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] #  weights in matrix multiplication and embedding - regularization - any single wt not large and distribute wt
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # biases and 1 D tensors; layer norms scales and biases
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay}, # embeddings and matmul participating wts
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # check if fused available - faster when running cuda, launch lot of kernels into one - single time kernel updates
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer




import tiktoken
import numpy as np
# Example of 1 batch
# enc = tiktoken.get_encoding('gpt2')
# with open('input.txt', 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T+1])
# buf = buf.to(device)
# x = buf[:-1].view(B,T)
# y = buf[1:].view(B,T)

enc = tiktoken.get_encoding('gpt2')

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# Data loader class
class DataLoaderLite:
    def __init__(self, B, T,process_rank,num_processes,split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ('train', 'val')

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self): # reset data loader as do model eval every 100th iteration
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank


    def next_batch(self):
        B, T = self.B, self.T
        buf = torch.tensor(self.tokens[self.current_position:self.current_position + B*T+1])

        x = buf[:-1].view(B,T) # inputs
        y = buf[1:].view(B,T) # targets
        # advance position in tensor
        self.current_position += B*T*self.num_processes # because now we process B*T*self.num_processes tokens each time
        # if loading next position out of bounds, load next batch
        if self.current_position + (B*T*self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# Get device
# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
#device = "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
elif torch.backends.mps.is_available():
    torch.manual_seed(1337)


# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=2 train_gpt2_weights_grad_dataset_validation.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # all processes have diff rank, every gpu has diff values ex. 0 to 7
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # multi node setting - rank of gpu on diff box
    ddp_world_size = int(os.environ['WORLD_SIZE']) # torch.run set environment variables, total number of processes running ex. 8
    device = f'cuda:{ddp_local_rank}' # which gpu to use
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else: # single gpu training
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")




total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 8 # micro batch size
T = 512 # sequence length
assert total_batch_size % (B*T * ddp_world_size) == 0, "make sure total batch size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process: # print single time
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}") # run backward and forward but not update gradients

print("I am GPU", ddp_rank)
# import sys; sys.exit(0)

train_loader = DataLoaderLite(B=8, T=512, process_rank=ddp_rank, num_processes=ddp_world_size, split = "train") # batch size equivalent to gpu usage, In number of tokens, batch size is 8*512. To inc batch size - use gradient accumulation - run longer and process multiple sequence and add gradients
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val") # Validation data loader

# every process to get its own chunk of data
torch.set_float32_matmul_precision('high') # enable internal precision of tensorflow32, matrix multiplication us tf32 precision - works on gpu a100

#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig()) # 2 GPT models across 2 GPUs
print('all worked')
# model.eval()
model.to(device)
model = torch.compile(model) # add compilation time but make faster; analyze model - not run layer by layer -> optimize process, compile NN as single object efficiently
# GPU has its own memory - where bandwidth is still limited-> within chip everything fast; torch compile optimize so less number of calls to memory - kernel fusion. 
# Memory on chip - l2 cache / l1 cache / register - limited memory on chip but latency extremely fast - input from hbm streamed to chip - calculation and store back to global memory
# with torch.compile -> while on chip -> data process stay on chip - fast operate on -> and single round trip back. 

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # act similar in forward pass, but in backward pass - each gpu has grad for params - call all reduce - avg grads across
    # all ranks and deposit on all ranks 
raw_model = model.module if ddp else model # contains "raw" unwrapped model
# logits, loss = model(x, y)
# print(loss) # 4, 32, 50257
# cross entropy loss = -log(1/vocab size) for random uniform = 10.2 expected loss 


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10 # 715; warmup over 375M tokens, 1B/375m = 715 
max_steps = 100 # 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens; 2^19 tokens per step -> and 10Billion tokens, max steps 19073 steps

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95),eps=1e-8) # match gpt3 paper
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

for step in range(max_steps):
    t0= time.time()
    # once in a while evaluate our validation loss
    if step % 50 == 0: # can compare the model wts from actual GPT2 here to see how our model compares to gpt2 on fine web data - but might have trained on diff data distribution so not fair comparison
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")

    # once in a while generate from the model (except step 0, which is noise)
    if step > 0 and step % 10 == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank) # sample generator separate from one used in training model.
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps): 
        x, y = train_loader.next_batch() # see easy gains ex. for token with very less usage. 
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # ON ALL ranks, avg of all ranks on all ranks
        with torch.autocast(device_type=device, dtype=torch.float16): # use float16 for Mac - not for GPU
            logits, loss = model(x, y)
            #import code; code.interact(local=locals()) - logits bfloat32 whereas model.transformer.wte.weight.dtype same as before
        loss = loss / grad_accum_steps # recovering the normalizer
        loss_accum += loss.detach() # detach tensor from graph
        #import code; code.interact(local=locals()) # by default, everything in float32 -> can lower precision here - number have fewer bits - move around - memory bandwidth increase.
        loss.backward() # accumulate gradients - syncronize at the last step only when micro step becomes grad_accum_steps use no_sync context managers
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients to 1.0 -> norm of param vector <= 1.0; unlucky in batch - high loss - high grad - shock model ; should be stable - start high as model random initially
    
    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step() # update gradients to reduce loss
    #torch.cuda.synchronize() # for gpu to dinish all tasks given by cpu
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000 # time diff in millisec
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1-t0) # how many tokens training per sec
    if master_process:
        print(f"step {step}, loss_accum: {loss_accum.item():.6f}, norm:{norm:.4f}, lr:{lr:.7f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}") # item get back to cpu
    #break

if ddp:
    destroy_process_group()

import sys; sys.exit()

# num_return_sequences = 5
# max_length = 30

# # prefix tokens
# enc =  tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# x = tokens.to(device)

# # generate, right now x is (B, T) where B = 5 and T = 8
# # set seed to 42
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# while x.size(1) < max_length:
#     # forward model to get logits
#     with torch.no_grad():
#         logits = model(x) # (b, t, vocab_size)
#         # take logits at the last position
#         logits = logits[:, -1, : ] # (b, vocab_size) - get last column logits 
#         # get probs
#         probs = F.softmax(logits, dim=-1)
#         # do top k sampling of 50
#         # topk probs here becomes (5, 50), topk_indices (5, 50) - not sample rare tokens - model on track.
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # select token from top-k probs
#         ix = torch.multinomial(topk_probs, 1) # (B, 1)
#         # gather corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         # append to sequence
#         x = torch.cat((x, xcol), dim=1)

# # print generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)