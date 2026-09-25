"""
Minimal decoder-only transformer, trained character by character, with a chat loop at the end.
No huggingface, no tiktoken. Everything built from scratch so you can see every piece.
Run: pip install torch --break-system-packages
Then: python minimal_chatbot.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 1. Tiny training corpus. Swap this out for a real text file if you want.
# ---------------------------------------------------------------------------
text = """
hello there
how are you doing today
i am doing well thank you for asking
what is your name
my name is astra
what can you help me with
i can help you learn about transformers
that sounds interesting
let us build something together
"""

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

# ---------------------------------------------------------------------------
# 2. Hyperparameters
# ---------------------------------------------------------------------------
block_size = 32      # how many tokens of context the model sees
batch_size = 16
n_embd = 64           # embedding dimension
n_head = 4
n_layer = 3
dropout = 0.1
learning_rate = 3e-4
max_iters = 3000
eval_interval = 300

def get_batch():
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# 3. Self-attention, built from the Q/K/V matrices directly
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # scaled dot product
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # residual connection around attention
        x = x + self.ffwd(self.ln2(x))  # residual connection around feedforward
        return x

# ---------------------------------------------------------------------------
# 4. Full model: embeddings + positional encoding + stacked blocks + output head
# ---------------------------------------------------------------------------
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.9):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            next_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_idx = sorted_idx.gather(-1, next_sorted_idx)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
def train():
    model = MiniGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            print(f"step {step}: loss {loss.item():.4f}")

    return model

# ---------------------------------------------------------------------------
# 6. Chat loop
# ---------------------------------------------------------------------------
def chat(model):
    print("\nmodel trained. type something and it will try to continue it. ctrl+c to quit.\n")
    while True:
        try:
            prompt = input("you: ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=100)
        print("model:", decode(out[0].tolist()))


if __name__ == "__main__":
    trained_model = train()
    chat(trained_model)