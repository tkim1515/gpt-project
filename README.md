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





















