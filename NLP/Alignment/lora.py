import copy
import math

import torch
from peft import LoraConfig, get_peft_model
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
            self,
            base: nn.Linear,
            rank: int = 4,
            alpha: float = 0.0
    ):
        super().__init__()

        self.base = base

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype
        )

        merged.weight.copy_(
            self.base.weight + self.scale * (self.B @ self.A)
        )

        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)

        return merged


def add_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0
) -> nn.Module:

    for p in model.parameters():
        p.requires_grad_(False)

    model.head = LoRALinear(
        model.head,
        rank=rank,
        alpha=alpha
    )

    return model

def add_lora_peft(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0
) -> nn.Module:
    config = LoraConfig(
        r=rank,
        lora_alpha=int(alpha),
        lora_dropout=0.0,
        target_modules=["head"],
        bias="none"
    )

    return get_peft_model(model, config)

class TinyLLM(nn.Module):
    def __init__(
            self,
            vocab_size: int = 64,
            hidden_size: int = 32
    ):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, hidden_size)

        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        x, _ = self.rnn(x)
        return self.head(x)

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def compare_lora():
    torch.manual_seed(42)

    base_model = TinyLLM()

    custom_model = add_lora(
        copy.deepcopy(base_model),
        rank=4,
        alpha=8
    )

    peft_model = add_lora_peft(
        copy.deepcopy(base_model),
        rank=4,
        alpha=8
    )

    print("Custom LoRA parameters:", count_trainable_parameters(custom_model))
    print("PEFT LoRA parameters:", count_trainable_parameters(peft_model))

    tokens = torch.randint(0, 64, (4, 10))

    with torch.no_grad():
        custom_logits = custom_model(tokens)
        peft_logits = peft_model(tokens)

    torch.testing.assert_close(custom_logits, peft_logits)

if __name__ == "__main__":
    compare_lora()

