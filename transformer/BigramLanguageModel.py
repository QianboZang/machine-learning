import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Block


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device):
        super().__init__()
        # vocab_size: the number of unique tokens in the vocabulary
        # n_embd: the dimension of the embedding
        # block_size: the number of tokens in a block
        # n_head: the number of heads in the multi-head attention
        # n_layer: the number of transformer blocks
        # device: the device on which the model will be trained
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x)    
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size):
        # idx: the starting sequence of tokens
        for _ in range(max_new_tokens):
            # get the last block of tokens
            idx_cond = idx[:, -block_size: ]
            # get the logits and loss
            logits, loss = self(idx_cond)
            # get the logits for the last token
            logits = logits[:, -1, : ] 
            # get the distribution over the vocabulary
            probs = F.softmax(logits, dim=-1)
            # sample the next token
            idx_next = torch.multinomial(probs, num_samples=1) 
            # add the next token to the sequence
            idx = torch.cat((idx, idx_next), dim=1) 
            # (B, T+1)
        return idx
    