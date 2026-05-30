# Shakespeare GPT 텍스트 생성 프로그램 설명

이 프로그램은 셰익스피어 작품 텍스트를 학습하여, 셰익스피어 문체와 유사한 새로운 문장을 생성하는 GPT 기반 텍스트 생성 프로그램이다.

---

## 1. 데이터 저장

```python
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_DATA_PATH = Path("input.txt")
DEFAULT_MODEL_PATH = Path("shakespeare_gpt.pt")
```

데이터를 `input.txt`에, 학습한 모델을 `shakespeare_gpt.pt`에 저장한다.

`Path`를 사용하면 `data_path.exists()` 등의 함수를 사용할 수 있다.

---

## 2. 데이터 확인

```python
def ensure_data(data_path: Path) -> None:
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, data_path)
```

데이터가 `data_path`에 저장되어있는지 확인하고, 없으면 데이터를 저장할 경로를 생성한 후 `DATA_URL`에서 데이터를 다운로드 받아 `data_path` 위치에 저장한다.

---

## 3. 데이터 읽기

```python
def load_text(data_path: Path) -> str:
    ensure_data(data_path)
    return data_path.read_text(encoding="utf-8")
```

2의 `ensure_data`를 사용해 `data_path`에 데이터를 확실히 저장하고 `read_text`로 `input.txt`의 전체 데이터를 반환한다.

---

## 4. `chars`, `stoi`, `itos` 정의

```python
def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos, len(chars)
```

`input.txt` 안의 글자 종류를 정렬하여 `chars`에 저장하고 그 길이를 반환하여 글자 종류 개수를 확인한다.

`stoi`는 특정 문자를 특정 숫자로, `itos`는 특정 숫자를 특정 문자로 저장한다.

---

## 5. 텍스트를 숫자로 변환

```python
def encode(text: str, stoi) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
```

`text` 안의 모든 글자를 숫자로 변환하여 정수 tensor 타입으로 반환한다.

---

## 6. 데이터셋 만들기

```python
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
```

`init`에서 `data`, `block_size`를 받아 저장한다.

`len`에서 전체 데이터셋의 길이를 반환한다.

`getitem`에서 `idx`에 맞게 `x`에 `blocksize` 길이의 문제를, `y`에 `x`를 한칸 옆으로 옮긴 `blocksize` 길이의 정답을 저장해 반환한다.

---

## 7. 단일 Head Attention 정의

```python
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
```

Pytorch에서 `nn.Module`을 상속받는다.

`emb_dim`, `head_size`, `block_size`, `dropout`을 받는다.

`init`에서 `linear`로 `self.key`, `self.query`, `self.value`로 `emb_dim` 차원의 input을 `head_size` 차원의 벡터값으로 바꾼다.

1로 채워진 `[block_size, block_size]` 크기의 행렬을 만든 후 `torch_trill`로 아래쪽 삼각형만 남게 하여 미래의 값을 보지 못하게 한다. `self.trill`로 사용한다.

`tril`은 학습되는 파라미터는 아니지만 모델과 함께 관리되어야 하는 텐서이기 때문에 `register_buffer` 형식을 사용한다.

`Dropout`은 학습할 때 일부 값을 랜덤하게 0으로 만드는데, 모델이 특정 정보에 너무 의존하지 않게 해서 과적합을 줄인다.

`foward`에서 실제로 `(batchsize, blocksize, emd_dim)` 형태의 input `x`를 받아 `self.key`, `self.query`, `self.value`로 `head_size` 차원의 벡터값으로 바꿔 `k`, `q`, `v`로 저장한다.

이때 `q`는 내가 목표로 하는 것이며 비슷한 `k`를 가질수록 더 많이 참고한다. 그리고 실제로는 `v`를 가져와서 `q`와 `k`를 비교해 나온 비율대로 섞는다.

`wei`는 이렇게 섞을 때의 가중치이다.

먼저 `q.shape = [B, T, head_size]`, `k.shape = [B, T, head_size]` 인데 `k.transpose(-2, -1)`는 마지막 두 차원을 뒤바꿔 `k.shape = [B, head_size, T]`으로 만들고 `q`와 행렬곱셈시켜 `[B, T, T]`의 형태로 만든다.

이렇게 만들어진 `wei`에 `k.size(-1)=block_size`에 `**-0.5`를 취하여 곱해준다. 이것은 값이 너무 커지는 것을 막아 결과가 한 곳에 몰리는 것을 방지해준다.

예측에서 미래의 값을 참고해선 안되므로 `self.tril`로 `[block_size, block_size]` 크기의 행렬을 만들고 위쪽 삼각형의 값을 모두 0으로 바꾸고 `masked_fill(, float("-inf"))`로 0을 모두 `-inf`로 바꿔준다. 이제 미래 값은 참고할 수 없게 된다.

`F.softmax`로 합이 1이되는 확률 형태로 바꾼다.

`self.dropout`으로 일부 비율을 랜덤하게 0으로 만든다.

최종적으로 `[B, T, T]` 형태의 `wei`에 `[B, T, head_size]` 형태의 `v`를 행렬곱셈하여 `[B, T, head_size]` 형태의 `out`을 만들어 반환한다.

이때 `out`은 확률이 아닌 특정 벡터값이다.

---

## 9. Multi Head Attention 정의

```python
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
```

`multi head attention`은 위에서 정의한 `head attention`을 병렬로 실행한 뒤 합친다.

`emb_dim`, `num_heads`, `block_size`, `dropout`을 받는다.

각 `head`의 결과를 합친 것이 `emb_dim`과 같아야 하므로 `head_size = emb_dim // num_heads`가 된다.

`self.heads`에 위에서 정의한 `Head` 클래스를 사용해 `num_heads`개의 `Head`를 만든다.

`forward`에서 `self.heads`에 나온 결과값들을 모두 합친다.

`self.proj`를 통해 한번 더 섞어서 조합한다.

`self.dropout`을 통해 dropout처리를 한 후 최총 `out`을 반환한다.

---

## 10. FeedForward

```python
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
```

`emb_dim` 크기의 input을 `4 * emb_dim` 크기로 늘린 후 다시 `emb_dim`으로 줄여 특징을 더 잘 표현하게 해준다.

---

## 11. Block

```python
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
```

`Block`은 위에서 정의한 `MultiHeadAttention`과 `FeedForward`를 합친 클래스이다.

`self.ln1`으로 input을 정규화하여 값이 너무 커지거나 작아지는 것을 방지한다.

`self.sa`로 `MultiHeadAttention`을 돌린다.

`self.ln2`로 input을 다르게 정규화하여 값이 너무 커지거나 작아지는 것을 방지한다.

`self.ffwd`로 `FeedForward`를 돌린다.

`forward`에서 `x`에 input을 정규화한 값을 `MultiHeadAttention`에 넣은 변화량을 더하여 새 `x`를 만든다.

이 방식으로 원래 정보를 유지하면서 학습이 안정적으로 진행된다.

새로운 `x`를 다시 정규화하여 `FeedForward`를 돌린다.

`x`에 `FeedForward`의 결과값인 변화량을 더하여 새 `x`를 만든다.

`Block`은 여러차례 반복되기에 input 형태와 output형태가 같은 것이 좋다.

여기에서도 input과 `out`의 형태가 같도록 유지된다.

---

## 12. Tiny GPT

```python
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
```

`B`, `T`에 batch size, block_size를 저장한다.

`torch.arange`로 각 토큰의 글자 위치를 기억하게 한다.

token의 값, 위치를 각각 embedding 한다.

위치를 embedding하면 `[T, emb_dim]` 형태가 되는데, `tok`와 더해주려 하므로 형태를 맞춰주기 위해 `[None]`으로 앞에 차원을 추가해 `[1, T, emb_dim]`의 형태로 만들어준다. 이제 broadcasting rule에 의해 계산 가능하다.

token의 값, 위치를 각각 embedding한 값을 더하여 `h`에 저장한다. 이때 `h`는 위치 정보와 값 정보를 모두 가지게 된다.

`h`를 여러 개의 `Block`에 통과시킨 후 정규화한다.

최종 `h`를 `[B, T, vocab_size]`으로 바꾼 `logits`를 반환한다.

이때 `logits`는 각 위치에서 다음 글자 후보들에 대한 점수이다.

---

## 13. Loss 정의

```python
def sequence_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets)
```

`TinyGPT`가 각 위치에서 예측한 다음 토큰 점수와 실제 정답 토큰을 비교해서 cross entropy loss를 계산한다.

`F.cross_entropy`는 `[B, vocab_size, T]` 형태의 input이 필요하므로 `logits` 형태의 indes 1, 2의 위치를 바꿔준다.

---

## 14. 훈련

```python
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
```

1 epoch만큼 훈련하는 함수. 추가로 학습하려면 `max_steps`를 늘리면 된다.

`model.train()`은 dropout을 활성화 시킨다.

`total_loss`, `total_count`에 전체 loss 합, 처리한 데이터 개수를 저장하기 위해 각각 `0.0`, `0`을 설정한다.

데이터 `loader`는 데이터셋에서 각각 `batch_size`개의 `x`, `y`를 꺼내 각각 `xb`, `yb`에 저장한다.

`xb`를 모델에 넣으면 모델은 다음 문자에 대한 후보점수를 계산해 `logits`에 저장한다.

`logits`에 저장된 예측과 정답 `yb`를 비교하여 loss를 계산한다.

grad를 0으로 만들어주고, loss를 바탕으로 grad를 역으로 다시 계산한 후, 파라미터들을 grad에 따라 수정해준다. (`token_embedding`, `position_embedding`, `key/query/value`, `feedforward linear layer`, `lm_head` 등)

`F.cross_entropy`는 평균 loss 값을 계산하므로, `loss.item`으로 loss를 일반 숫자 형태로 만들어준 후 `xb.size(0)=batch_size`를 곱해 전체 loss 값을 계산하여 `total_loss`에 누적한다.

batch를 계산한 수만큼 `xb.size(0)=batch_size`를 더해 처리한 데이터 개수를 저장한다.

한번 할때마다 `step`에 1을 더하고 `step`이 `max_step`에 도달하면 멈춘다.

`total_loss`를 `total_count`로 나누어 평균 loss를 반환한다.

---

## 15. 생성

```python
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
```

이 단계는 실제 결과를 생성하는 단계이므로 `@torch.no_grad()`로 grad를 계산하지 않겠다고 표현한다.

`model.eval()`으로 dropout를 제거한다.

`context`는 모델에 넣은 입력값. `(1, block_size)` 형태의 0으로 이루어진 행렬로 시작한다.

`start_text`의 모든 글자에 대하여 `stoi`에 있으면 차례로 숫자로 바꿔 `context` 뒤쪽에 추가하고 `context` 앞쪽은 버리는 작업을 해준다.

그리고 출력값이 저장될 `out` 리스트에 `start_text`를 먼저 넣어준다.

`max_new_tokens`번 만큼 model에 context를 넣고 context의 다음 글자의 후보점수를 예측한 것을 확률 형태로 바꾸어 확률에 따라 임의로 추출하여 일반 숫자 형태로 바꾼 후 글자로 바꾸어 `out`에 추가하고 context를 한깐 옆으로 밀는 작업을 반복한다.

`logits`는 model을 거쳤을 때 `[1, block_size, vocab_size]` 형태이지만 마지막의 예측치만이 다음 글자의 후보점수이므로 마지막 예측치만 가져와 `[1, vocab_size]` 형태로 바꿔준다.

최종적으로 리스트 안의 글자들을 모두 합쳐 출력물을 반환한다.

---

## 16. 모델 저장

```python
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
```

학습한 모델을 나중에 사용할 수 있게 저장한다.

`torch.save`로 `path`에 모델 관련 정보를 저장한다.

`model.state_dict()`은 모델이 학습한 숫자값들, 즉 가중치, bias를 의미한다.

`config`는 `vocab_size`, `block_size` 등 모델 설정 값이다.

---

## 17. 모델 불러오기

```python
def load_model(path):
    payload = torch.load(path, map_location="cpu")
    config = payload["config"]
    model = TinyGPT(**config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload["stoi"], payload["itos"], config
```

`payload`에 앞에서 `path`에 저장한 정보들을 불러와 model, `stoi`, `itos`, `config`를 다시 만든다.

`**config`는 config에 저장된 `vocab_size: 65` 등을 `vocab_size=65`로 만든다. 이것을 `TinyGPT`에 넣어 모델의 틀을 재구성한다.

`model.state_dict()`에 저장된 가중치들을 모델에 넣어 학습된 모델로 바꿔준다.

`model.eval()`로 dropout을 꺼준다.

---

## 17. 옵션 저장

```python
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
```

터미널에서 사용할 옵션과 디폴트값 정리.

---

## 18. 프로그램 실행

```python
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
```

`args = parse_args()`으로 터미널에 입력된 옵션을 읽고 그에 맞춰 실행한다.

`Path`로 데이터와 모델 경로 설정한다.

사용자의 옵션에 따라 훈련/생성 실행. 옵션이 없을 경우 모델이 있으면 생성, 없으면 학습 후 생성한다.

훈련할 경우 먼저 데이터 불러오기.

모델이 존재하고 이어서 학습하는 경우 모델을 불러와 기존의 `block_size`, `stoi`를 사용한다.

데이터를 `stoi`로 숫자로 변환하고 새로 데이터셋과 데이터로더를 만들어 학습한다.

새로 학습하는 경우 처음부터 새로 제작한다.

지정한 epoch 수만큼 `train_one_epoch`을 사용하여 학습하고 epoch과 loss 출력, 모델 저장한다.

생성하는 경우 모델 파일 없으면 오류, 있으면 모델을 불러온다.

학습된 모델로 텍스트 생성, 출력한다.

---

## 19. 실행방법

```python
if __name__ == "__main__":
    main()
```

터미널에 직접 `shakespeare_gpt.py`가 실행된 경우에만 `main` 실행한다.

`import` 시 실행되는 것을 방지해준다.
