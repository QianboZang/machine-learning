import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import tiktoken
from BigramLanguageModel import BigramLanguageModel
import wandb


def main():
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # hyper parameters
    batch_size = 128 
    block_size = 64 # block
    max_iters = 5000
    learning_rate = 1e-3
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    eval_iters = 1000
    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.0
    print(device)

    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": max_iters,
        "batch_size": batch_size
    }

    with open('/home/zang/language_model/01_transformer/49010-0.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    enc = tiktoken.get_encoding("gpt2")
    assert enc.decode(enc.encode("hello world")) == "hello world"
    # encoder: take a string, output a list of integers -> enc.encode(text)
    # decoder: take a list of integers, output a string -> enc.decode(number)
    print(enc.encode("hello world"))
    print(enc.decode(enc.encode("hello world")))
    vocab_size = enc.n_vocab

    data = torch.tensor(enc.encode(text), dtype=torch.long)
    print(data.shape, data.dtype)
    print(data[: 100]) 

    n = int(0.9 * len(data)) # first 90% will be train, rest val
    train_data = data[: n]
    val_data = data[n: ]

    model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout, device).to(device)
    wandb.watch(model)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # data loading
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i: i+block_size] for i in ix])
        y = torch.stack([data[i+1: i+block_size+1] for i in ix])
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
    
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % 500 == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # generate from the model
            context_example = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated_text = enc.decode(model.generate(context_example, 2000, block_size)[0].tolist())
            wandb.log({
                "train_loss": losses['train'], "val_loss": losses['val'], 
                "generated_text": wandb.Html(generated_text)
            })

        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item()})
            
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(enc.decode(model.generate(context, 2000, block_size)[0].tolist()))


if __name__ == "__main__":
    wandb.init(project="language", entity="dl_prac")
    main()
