from typing import Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class SyntheticCLIPDataset(Dataset):
    def __init__(self, num_samples: int = 500, num_classes: int = 10):
        self.images = torch.randn(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.texts = [f"class {int(label)}" for label in self.labels]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        return self.images[idx], self.texts[idx]


def tokenize(texts, max_len: int = 4) -> torch.Tensor:
    tokenized = []
    for text in texts:
        tokens = text.split()
        ids = [int(tok) if tok.isdigit() else 0 for tok in tokens]
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        tokenized.append(ids)
    return torch.tensor(tokenized)

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self.net(X)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 100, embed_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self.embedding(X)
        return x.mean(dim=1)

class CLIPAdapter(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class CLIP(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.image_adapter = CLIPAdapter(embed_dim)
        self.text_adapter = CLIPAdapter(embed_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, images, text_tokens) -> torch.Tensor:
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(text_tokens)

        img_emb = self.image_adapter(img_emb)
        txt_emb = self.text_adapter(txt_emb)

        img_emb = F.normalize(img_emb, dim=1)
        txt_emb = F.normalize(txt_emb, dim=1)

        logits = img_emb @ txt_emb.T / self.temperature
        return logits

def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.size(0))
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

def train_clip() -> nn.Module:
    dataset = SyntheticCLIPDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CLIP(embed_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}")

        for images, texts in progress_bar:
            tokens = tokenize(texts)

            logits = model(images, tokens)
            loss = clip_loss(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"[Scratch CLIP] Epoch {epoch + 1}: avg loss = {avg_loss:.4f}")

    return model

def evaluate_scratch_clip(model: nn.Module) -> None:
    model.eval()

    images = torch.randn(4, 3, 32, 32)
    texts = ["class 0", "class 1", "class 2", "class 3"]
    tokens = tokenize(texts)

    with torch.no_grad():
        logits = model(images, tokens)

    print("\nSimilarity matrix (Scratch CLIP):")
    print(logits)

def transformers_clip() -> None:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    images = torch.rand(2, 3, 224, 224)
    texts = ["a photo of a cat", "a photo of a dog"]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    print("\nSimilarity matrix (Transformers CLIP):")
    print(outputs.logits_per_image)

if __name__ == "__main__":
    scratch_clip = train_clip()
    evaluate_scratch_clip(scratch_clip)
    transformers_clip()

