import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os

# conf
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
out_dir = 'swarmAtt_checkpoints'
always_save_checkpoint = False # if True, always save a checkpoint after each eval

# adamw optimizer
learning_rate = 2e-4 # max learning rate
max_iters = 3000 # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.0 # 1.0 # clip gradients at this value, or disable if == 0.0
dropout = 0.2

eval_iters = 200 # how many batches to evalute loss over
eval_interval = 200 # how often to evaluate the loss
"""
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 200 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate/10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
"""

# hyperparameters
batch_size = 64 # 64 # how many independent sequences will we process in parallel?
block_size = 256 # 256 # what is the maximum context length for predictions?
n_embd = 64*4 # 64*4 
n_head = 8 # 4
n_layer = 14 # 16
#inf_steps = list(range(1,5)) + [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 720] # try 1,2,3,8,16,32
#inf_steps = range(1,n_head+1) # stride in the influence in the different heads
inf_steps = [1,2,3,4,1,2,3,4]
head_steps = 1 # number of repeated steps of influence in one head
# ------------

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))] # only save hyperparameters that are int, float, bool, str or list
config = {k: globals()[k] for k in config_keys} # for logging
# -----------------------------------------------------------------------------

torch.manual_seed(1337)

input_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of custom attention """

    def __init__(self, head_size, shift = 1):
        super().__init__()
        self.identity = nn.Linear(n_embd, head_size, bias=False)
        self.influence = nn.Linear(n_embd, head_size, bias=False)
        self.Khead = nn.Parameter(torch.tensor([0.5]))
        #self.mass = nn.Linear(n_embd, 1, bias=False)
        self.shift = shift

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        id = self.identity(x)   # (B,T,hs)
        inf = self.influence(x) # (B,T,hs)
        #K = F.sigmoid(self.Khead)
        K = torch.exp(self.Khead)
        # adapt identities
        inf = self.dropout(inf)
        for t in range(head_steps): # number of steps in the head
            B, T, hs = inf.shape
            zeros = torch.zeros(B, min(self.shift,T), hs, device=inf.device, dtype=inf.dtype) # adapting zeros size for inference
            inf = 1/(1+K) * torch.cat([zeros, inf[:, :-self.shift, :]], dim=-2)
            id = K/(1+K) * id + inf # (B,T,hs)
        out = id

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size,shift) for shift in inf_steps[:num_heads]]) # different shift for each head
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

    
class MultiHeadSwarmAttention(nn.Module):
    """ implements multiple heads of swarm attention in parallel in an efficient way """

    def __init__(self, num_heads, head_size):
        super().__init__()
        assert n_embd % num_heads == 0 # make sure n_embd is divisible by num_heads
        self.num_heads = num_heads
        self.head_size = head_size

        self.identity_all = nn.Linear(n_embd, n_embd, bias=False)
        self.influence_all = nn.Linear(n_embd, n_embd, bias=False)
        
        self.Khead = nn.Parameter(torch.ones(num_heads) * 0.5)
        self.shifts = inf_steps[:num_heads]
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        B, T, C = x.shape
        
        # Perform single, large matrix multiplications
        identities = self.identity_all(x)
        influences = self.influence_all(x)
        
        # Reshape to separate heads
        identities = identities.view(B, T, self.num_heads, self.head_size)
        influences = influences.view(B, T, self.num_heads, self.head_size)
        
        # Apply the head-specific logic
        K = torch.exp(self.Khead).view(1, 1, self.num_heads, 1)
        
        for t in range(head_steps):
            # A single operation for all shifts
            # Still sequential in nature, but the linear layers are parallel
            shifted_influences = torch.zeros_like(influences)
            influences = 1/(1+K) * influences
            # Shift each head's influences according to its specific shift value
            for i in range(self.num_heads):
                shift = self.shifts[i]
                shifted_influences[:, shift:, i, :] = self.dropout(influences[:, :-shift, i, :])
            influences = shifted_influences # Update influences for next iteration if needed
            identities = K / (1 + K) * identities + shifted_influences

        # Flatten the heads back together
        out = identities.view(B, T, C)
        
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        #self.sa = MultiHeadAttention(n_head, head_size)
        self.sa = MultiHeadSwarmAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.sa(self.ln1(x)) + x # residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        #pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb #+ pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  vocab_size=None, dropout=dropout)


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

start_time = time.perf_counter()
best_val_loss = 1e9
logs = []
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        curr_time = time.perf_counter() - start_time
        one_log = f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {curr_time:.2f} s"
        print(one_log)
        logs.append(one_log)
        # save the model if the validation loss is the best we've seen so far
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                out_path = os.path.join(os.path.dirname(__file__), out_dir, 'ckpt.pt')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(checkpoint, out_path)

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # reset gradients
    loss.backward()
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

# generate from the saved (best) model

# Load checkpoint
print('Loading the best checkpoint')
ckpt_path = os.path.join(os.path.dirname(__file__), out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

# Save a txt log (edit to make unique filenames later)
def save_training_log(logs, config, filename="training_log.txt"):
    log_dir = os.path.join(os.path.dirname(__file__), out_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    with open(log_path, "w") as f:
        for entry in logs:
            f.write(entry + "\n")
        f.write("\nModel Hyperparameters (config):\n")
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

save_training_log(logs,config)
