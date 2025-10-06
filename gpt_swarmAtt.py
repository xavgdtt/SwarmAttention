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
n_head = 8 # 8
n_layer = 14 # 14
#inf_steps = list(range(1,5)) + [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 720] # try 1,2,3,8,16,32
#inf_steps = range(1,n_head+1) # stride in the influence in the different heads
inf_steps = [1,2,3,4,1,2,3,4]
head_steps = 1 # number of repeated steps of influence in one head
formation_loss_weight = 0.05 # weight of the formation loss in the total loss
formation_target_distance = 0 # target distance for the formation loss
# 1 (tested and does not improve wrt to without, should I set them to be close? like 1e-3?) 
# 0.001 (tested and improves wrt to without, when loss is computed after attention and before residual)
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
    out_main_losses = {}
    out_formation_losses = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        main_losses = torch.zeros(eval_iters)
        formation_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss, main_loss, formation_loss = model(X, Y)
            # record losses separately for monitoring
            losses[k], main_losses[k], formation_losses[k] = loss.item(), main_loss.item(), formation_loss.item()
        out[split], out_main_losses[split], out_formation_losses[split] = losses.mean(), main_losses.mean(), formation_losses.mean()
    model.train()
    return out, out_main_losses, out_formation_losses

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

    def __init__(self, num_heads, head_size, target_distance=formation_target_distance, min_similarity=0.1): #0.2
        super().__init__()
        assert n_embd % num_heads == 0 # make sure n_embd is divisible by num_heads
        self.num_heads = num_heads
        self.head_size = head_size
        self.target_distance = target_distance # target distance for the formation loss
        self.min_similarity = min_similarity # minimum similarity to prevent collapse

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
    
    def compute_formation_loss(self, embeddings):
        """
        Computes a loss that encourages neighboring token embeddings to maintain
        a specific Euclidean distance.

        This acts as a geometric regularizer. It should be computed on the
        output of the layer and added to the main training loss.
        
        Args:
            embeddings (torch.Tensor): The output embeddings of the shape
                                       (batch_size, seq_len, head_size).

        Returns:
            torch.Tensor: A scalar loss value.
        """
        # Calculate squared Euclidean distances between adjacent tokens
        #shifted_embeddings = F.pad(embeddings, (0, 0, 1, 0))[:, :-1, :] # shift right by 1
        #distances_sq = ((embeddings - shifted_embeddings) ** 2).sum(dim=-1) # (B, T)
        
        #loss = torch.var(distances_sq[:, 1:])

        # The loss is the mean squared error between the actual squared
        # distances and the target squared distance.
        #loss = F.mse_loss(distances_sq[:, 1:], torch.full_like(distances_sq[:, 1:], self.target_distance**2)) # ignore first token
        
        # PRIMARY OBJECTIVE: Minimize variance of similarities
        # This encourages consistent relationships without fixing what they should be
        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        similarities = torch.sum(emb_norm[:, :-1] * emb_norm[:, 1:], dim=-1)  # (B, T-1)
        similarity_variance = torch.var(similarities, dim=-1).mean()  # Average variance across batch
        collapse_penalty = torch.relu(self.min_similarity - similarities.mean())
        loss = similarity_variance + 0.0 * collapse_penalty # 0.1 add penalty to prevent collapse

        #loss = loss / self.head_size # normalize by head size to keep loss scale consistent
        loss = loss / self.num_heads # normalize by number of heads to keep loss scale consistent

        return loss

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
        #self.sa = ApexSwarmAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        att_x = self.sa(self.ln1(x))
        formation_loss = self.compute_formation_loss(att_x) # compute formation loss on the attention output
        x = att_x + x # residual connection
        #formation_loss = self.compute_formation_loss(x)
        x_ln2 = self.ln2(x)
        #formation_loss = self.compute_formation_loss(x_ln2)
        x = x + self.ffwd(x_ln2)
        #formation_loss = self.compute_formation_loss(x)
        return x, formation_loss
    
    def compute_formation_loss(self, x):
        # Delegate to the attention layer's formation loss computation
        if hasattr(self.sa, 'compute_formation_loss'):
            return self.sa.compute_formation_loss(x)
        else:
            return torch.tensor(0.0).to(x.device)

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # use ModuleList to access individual blocks

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

        # Step 2: Manually iterate through the blocks
        total_formation_loss = torch.tensor([0.0]).to(device) # initialize formation loss
        for block in self.blocks:
            x, formation_loss = block(x)
            # If the block is a Swarm Block, compute and accumulate its formation loss
            if hasattr(block, 'compute_formation_loss'): # check if the method exists
                total_formation_loss += formation_loss
                

        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
            main_loss = None
            total_formation_loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            main_loss = F.cross_entropy(logits, targets)

            # Step 3: Combine the main loss with the formation loss
            # The weight is a crucial hyperparameter to balance the two objectives.
            loss = main_loss + formation_loss_weight * total_formation_loss

        return logits, loss, main_loss, total_formation_loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits,_,_,_ = self(idx_cond)
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
        losses, main_losses, formation_losses = estimate_loss()
        curr_time = time.perf_counter() - start_time
        one_log = (
            f"step {iter}: "
            f"train loss {losses['train']:.4f} "
            f"(main: {main_losses['train']:.4f}, form: {formation_losses['train']:.4f}), "
            f"val loss {losses['val']:.4f} "
            f"(main: {main_losses['val']:.4f}, form: {formation_losses['val']:.4f}), "
            f"time {curr_time:.2f} s"
        )
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
    logits, loss, main_loss, formation_loss = model(xb, yb)
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
