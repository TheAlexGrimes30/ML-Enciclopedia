from typing import Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class SyntheticCLIPDataset(Dataset):
    """
    Synthetic dataset for training a CLIP-like model.
    """

    def __init__(self, num_samples: int = 500, num_classes: int = 10):
        """
        Constructor of synthetic dataset.

        :param num_samples: Number of samples in the dataset
        :param num_classes: Number of distinct classes
        """

        self.images = torch.randn(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.texts = [f"class {int(label)}" for label in self.labels]

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        """

        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        """
        Retrieve a single sample from the dataset.

        :param idx: Index of the sample
        :return: Tuple of image tensor and corresponding text
        """

        return self.images[idx], self.texts[idx]


def tokenize(texts, max_len: int = 4) -> torch.Tensor:
    """
    Tokenizer for synthetic text inputs.

    Converts text like "class 3" into a sequence of integers.
    This tokenizer is intentionally naive and used only for demonstration.

    :param texts: List of input text strings
    :param max_len: Maximum sequence length
    :return: Tensor of token IDs with shape (batch_size, max_len)
    """

    tokenized = []
    for text in texts:
        tokens = text.split()
        ids = [int(tok) if tok.isdigit() else 0 for tok in tokens]
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        tokenized.append(ids)
    return torch.tensor(tokenized)

class ImageEncoder(nn.Module):
    """
    Convolutional image encoder.

    Encodes an image into a fixed-size embedding vector.
    """

    def __init__(self, embed_dim: int = 128):
        """
        Constructor of image encoder.

        :param embed_dim: Dimensionality of the output embedding
        """

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
        """
        Forward pass of the image encoder.

        :param X: Input image tensor of shape (batch_size, 3, H, W)
        :return: Image embeddings of shape (batch_size, embed_dim)
        """

        x = self.net(X)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TextEncoder(nn.Module):
    """
    Text encoder based on token embeddings.

    Computes sentence embeddings by averaging token embeddings.
    """

    def __init__(self, vocab_size: int = 100, embed_dim: int = 128):
        """
        Constructor of text encoder.

        :param vocab_size: Size of the token vocabulary
        :param embed_dim: Dimensionality of the output embedding
        """

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the text encoder.

        :param X: Token IDs of shape (batch_size, seq_len)
        :return: Text embeddings of shape (batch_size, embed_dim)
        """

        x = self.embedding(X)
        return x.mean(dim=1)

class CLIPAdapter(nn.Module):
    """
    Projection adapter used to refine embeddings.

    Acts as a small MLP projection head similar to those used
    in contrastive learning frameworks.
    """

    def __init__(self, embed_dim: int):
        """
        Constructor of adapter.

        :param embed_dim: Dimensionality of input and output embeddings
        """

        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the adapter.

        :param x: Input embeddings
        :return: Projected embeddings
        """

        return self.proj(x)

class CLIP(nn.Module):
    """
    CLIP-like model combining image and text encoders.

    Produces similarity logits between image and text embeddings
    using cosine similarity with temperature scaling.
    """

    def __init__(self, embed_dim: int = 128, t: float = 0.07):
        """
        Constructor CLIP model.

        :param embed_dim: Shared embedding dimensionality
        :param t: Temperature
        """

        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.image_adapter = CLIPAdapter(embed_dim)
        self.text_adapter = CLIPAdapter(embed_dim)
        self.temperature = nn.Parameter(torch.tensor(t))

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CLIP model.

        :param images: Image tensor of shape (batch_size, 3, H, W)
        :param text_tokens: Tokenized text tensor
        :return: Similarity logits matrix (batch_size, batch_size)
        """

        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(text_tokens)

        img_emb = self.image_adapter(img_emb)
        txt_emb = self.text_adapter(txt_emb)

        img_emb = F.normalize(img_emb, dim=1)
        txt_emb = F.normalize(txt_emb, dim=1)

        logits = img_emb @ txt_emb.T / self.temperature
        return logits

def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the symmetric CLIP contrastive loss.

    Applies cross-entropy loss for both image-to-text
    and text-to-image directions.

    :param logits: Similarity matrix produced by the CLIP model
    :return: Scalar loss value
    """

    labels = torch.arange(logits.size(0))
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

def train_clip() -> nn.Module:
    """
    Train the scratch CLIP model on the synthetic dataset.

    :return: Trained CLIP model
    """

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
    """
    Evaluate the trained scratch CLIP model on random inputs.

    Prints the similarity matrix between images and texts.
    """

    model.eval()

    images = torch.randn(4, 3, 32, 32)
    texts = ["class 0", "class 1", "class 2", "class 3"]
    tokens = tokenize(texts)

    with torch.no_grad():
        logits = model(images, tokens)

    print("\nSimilarity matrix (Scratch CLIP):")
    print(logits)

def transformers_clip() -> None:
    """
    Demonstrate similarity computation using a pretrained CLIP
    model from Hugging Face Transformers.
    """

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

