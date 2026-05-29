import argparse
from pathlib import Path
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_DATA_PATH = Path("input.txt")
DEFAULT_MODEL_PATH = Path("shakespeare_gpt.pt")


def ensure_data(data_path: Path) -> None:
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, data_path)


def load_text(data_path: Path) -> str:
    ensure_data(data_path)
    return data_path.read_text(encoding="utf-8")


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos, len(chars)


def encode(text: str, stoi) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


class NextTokenDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


class Head(nn.Module):
    def __init__(self, emb_dim: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        emb_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.Sequential(
            *[Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        tok = self.token_embedding(x)
        pos = self.position_embedding(pos)[None]
        h = tok + pos
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits


def sequence_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets)


def train_one_epoch(model, loader, optimizer, max_steps=None):
    model.train()
    total_loss, total_count = 0.0, 0
    for step, (xb, yb) in enumerate(loader):
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if max_steps is not None and step + 1 >= max_steps:
            break
    return total_loss / total_count


@torch.no_grad()
def sample_gpt(model, block_size, stoi, itos, start_text="ROMEO:", max_new_tokens=400):
    model.eval()
    context = torch.zeros((1, block_size), dtype=torch.long)
    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]])
            context = torch.cat([context[:, 1:], ix], dim=1)
    out = list(start_text)
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        out.append(itos[ix.item()])
        context = torch.cat([context[:, 1:], ix], dim=1)
    return "".join(out)


def save_model(path, model, config, stoi, itos):
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        },
        path,
    )


def load_model(path):
    payload = torch.load(path, map_location="cpu")
    config = payload["config"]
    model = TinyGPT(**config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload["stoi"], payload["itos"], config


def parse_args():
    parser = argparse.ArgumentParser(description="Train or generate TinyGPT on Shakespeare (CPU only).")
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--start_text", default="ROMEO:")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)

    do_train = args.train
    do_generate = args.generate
    if not do_train and not do_generate:
        if model_path.exists():
            do_generate = True
        else:
            do_train = True
            do_generate = True

    if do_train:
        text = load_text(data_path)
        if args.resume and model_path.exists():
            model, stoi, itos, config = load_model(model_path)
            block_size = config["block_size"]
            data = encode(text, stoi)
            dataset = NextTokenDataset(data, block_size)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        else:
            stoi, itos, vocab_size = build_vocab(text)
            data = encode(text, stoi)
            dataset = NextTokenDataset(data, args.block_size)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            config = {
                "vocab_size": vocab_size,
                "block_size": args.block_size,
                "emb_dim": args.emb_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            }
            model = TinyGPT(**config)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, loader, optimizer, max_steps=args.max_steps)
            print(f"epoch {epoch:2d} | train loss {train_loss:.4f}")

        save_model(model_path, model, config, stoi, itos)
        print(f"saved model to {model_path}")

    if do_generate:
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        model, stoi, itos, config = load_model(model_path)
        block_size = config["block_size"]
        text = sample_gpt(
            model,
            block_size,
            stoi,
            itos,
            start_text=args.start_text,
            max_new_tokens=args.max_new_tokens,
        )
        print(text)


if __name__ == "__main__":
    main()
