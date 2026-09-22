import math
import random
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42):
    random.seed(42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

@dataclass
class BertConfig:
    vocab_size: int

    hidden_size: int = 128
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 512

    max_position_embeddings: int = 128
    type_vocab_size: int = 2

    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    layer_norm_eps: float = 1e-12

class Tokenizer:
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MASK_TOKEN = "[MASK]"

    def __init__(self):
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.CLS_TOKEN,
            self.SEP_TOKEN,
            self.MASK_TOKEN,
        ]

        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text: str) -> list[str]:
        pattern = (
            r"\[(?:PAD|UNK|CLS|SEP|MASK)\]"
            r"|[\w]+"
            r"|[^\w\s]"
        )

        tokens = re.findall(
            pattern,
            text,
            flags=re.UNICODE
        )

        result = []

        for token in tokens:
            if token in self.special_tokens:
                result.append(token)
            else:
                result.append(token.lower())

        return result

    def build_vocab(self, texts: list[str]):
        vocab = set()

        for text in texts:
            vocab.update(self.tokenize(text))

        tokens = self.special_tokens + sorted(vocab)

        self.token_to_id = {
            token: idx
            for idx, token in enumerate(tokens)
        }

        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @property
    def pad_token_id(self):
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_token_id(self):
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def cls_token_id(self):
        return self.token_to_id[self.CLS_TOKEN]

    @property
    def sep_token_id(self):
        return self.token_to_id[self.SEP_TOKEN]

    @property
    def mask_token_id(self):
        return self.token_to_id[self.MASK_TOKEN]

    def convert_tokens_to_ids(
            self,
            tokens: list[str]
    ) -> list[int]:

        return [
            self.token_to_id.get(
                token,
                self.unk_token_id
            )
            for token in tokens
        ]

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [
            self.id_to_token.get(
                int(idx),
                self.UNK_TOKEN
            )
            for idx in ids
        ]

    def encode(
            self,
            text_a: str,
            text_b: Optional[str] = None,
            max_length: int = 128
    ):
        tokens_a = self.tokenize(text_a)

        tokens_b = None

        if text_b is not None:
            tokens_b = self.tokenize(text_b)

        if tokens_b is None:
            max_tokens = max_length - 2
            tokens_a = tokens_a[:max_tokens]

        else:
            max_tokens = max_length - 3

            while len(tokens_a) + len(tokens_b) > max_tokens:
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        tokens = [self.CLS_TOKEN]
        token_type_ids = [0]

        tokens.extend(tokens_a)
        token_type_ids.extend([0] * len(tokens_a))

        tokens.append(self.SEP_TOKEN)
        token_type_ids.append(0)

        if tokens_b is not None:
            tokens.extend(tokens_b)
            token_type_ids.extend([1] * len(tokens_b))

            tokens.append(self.SEP_TOKEN)
            token_type_ids.append(1)

        input_ids = self.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)

        input_ids.extend(
            [self.pad_token_id] * padding_length
        )

        token_type_ids.extend(
            [0] * padding_length
        )

        attention_mask.extend(
            [0] * padding_length
        )

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size
        )

        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size
        )

        self.token_type_embeddings = nn.Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size
        )

        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None
    ):
        batch_size, seq_length = input_ids.shape

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = (
            torch.arange(
                seq_length,
                device=input_ids.device
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                seq_length
            )
        )

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(
            config.hidden_size,
            self.all_head_size
        )

        self.key = nn.Linear(
            config.hidden_size,
            self.all_head_size
        )

        self.value = nn.Linear(
            config.hidden_size,
            self.all_head_size
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(
            self,
            x: torch.Tensor
    ):
        batch_size, seq_length, _ = x.shape

        x = x.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        )

        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        attention_scores = torch.matmul(
            query,
            key.transpose(-1, -2)
        )

        attention_scores = attention_scores * self.scale

        if attention_mask is not None:
            attention_scores = (
                attention_scores.masked_fill(
                    attention_mask == 0,
                    -1e4
                )
            )

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)

        context = context.permute(
            0,
            2,
            1,
            3
        ).contiguous()

        batch_size, seq_length, _, _ = context.shape

        context = context.view(
            batch_size,
            seq_length,
            self.all_head_size
        )

        return (
            context,
            attention_probs
        )

class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            input_tensor: torch.Tensor
    ) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()

        self.self_attention = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ):
        self_output, attention_probs = (
            self.self_attention(
                hidden_states,
                attention_mask
            )
        )

        attention_output = self.output(self_output, hidden_states)

        return (
            attention_output,
            attention_probs
        )

class GELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return 0.5 * x * (
            1.0
            + torch.erf(
                x / math.sqrt(2.0)
            )
        )

class BertIntermediate(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            config.intermediate_size
        )

        self.activation = GELU()

    def forward(
            self,
            hidden_states: torch.Tensor
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class BertOutput(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.dense = nn.Linear(
            config.intermediate_size,
            config.hidden_size
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            input_tensor: torch.Tensor
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states

class BertLayer(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ):
        attention_output, attention_probs = (
            self.attention(
                hidden_states,
                attention_mask
            )
        )

        intermediate_output = self.intermediate(attention_output)

        layer_output = self.output(
            intermediate_output,
            attention_output
        )

        return (
            layer_output,
            attention_probs
        )

class BertEncoder(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                BertLayer(config)
                for _ in range(
                    config.num_hidden_layers
                )
            ]
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ):
        attentions = []

        for layer in self.layers:
            hidden_states, attention_probs = (
                layer(
                    hidden_states,
                    attention_mask
                )
            )

            if output_attentions:
                attentions.append(
                    attention_probs
                )

        return (
            hidden_states,
            attentions
        )

class BertPooler(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size
        )

        self.activation = nn.Tanh()

    def forward(
            self,
            hidden_states: torch.Tensor
    ):
        cls_token = hidden_states[:, 0]

        pooled_output = self.dense(cls_token)
        pooled_output = self.activation(pooled_output)

        return pooled_output

class BertModel(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config

        self.embeddings = BertEmbeddings(
            config
        )

        self.encoder = BertEncoder(
            config
        )

        self.pooler = BertPooler(
            config
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ):
        embedding_output = self.embeddings(
            input_ids,
            token_type_ids
        )

        if attention_mask is not None:
            extended_attention_mask = (
                attention_mask[
                    :,
                    None,
                    None,
                    :
                ]
            )
        else:
            extended_attention_mask = None

        sequence_output, attentions = (
            self.encoder(
                embedding_output,
                extended_attention_mask,
                output_attentions
            )
        )

        pooled_output = self.pooler(sequence_output)

        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
            "attentions": attentions
        }


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size
        )

        self.activation = GELU()

        self.layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class BertMLMHead(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.transform = BertPredictionHeadTransform(config)

        self.decoder = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        self.bias = nn.Parameter(
            torch.zeros(
                config.vocab_size
            )
        )

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        logits = self.decoder(hidden_states)
        logits = logits + self.bias
        return logits

class BertNSPHead(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.classifier = nn.Linear(
            config.hidden_size,
            2
        )

    def forward(self, pooled_output):
        return self.classifier(pooled_output)

class BertForPreTraining(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()

        self.config = config

        self.bert = BertModel(config)
        self.mlm_head = BertMLMHead(config)
        self.nsp_head = BertNSPHead(config)

        self.mlm_head.decoder.weight = (
            self.bert
            .embeddings
            .word_embeddings
            .weight
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            mlm_labels: Optional[torch.Tensor] = None,
            nsp_labels: Optional[torch.Tensor] = None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs["pooler_output"]

        prediction_logits = self.mlm_head(
            sequence_output
        )

        sequence_relationship_logits = self.nsp_head(
            pooled_output
        )

        total_loss = None
        mlm_loss = None
        nsp_loss = None

        if mlm_labels is not None:
            mlm_loss = F.cross_entropy(
                prediction_logits.view(
                    -1,
                    self.config.vocab_size
                ),
                mlm_labels.view(-1),
                ignore_index=-100
            )

        if nsp_labels is not None:
            nsp_loss = F.cross_entropy(
                sequence_relationship_logits,
                nsp_labels
            )

        if (
            mlm_loss is not None
            and nsp_loss is not None
        ):
            total_loss = (
                mlm_loss
                + nsp_loss
            )

        elif mlm_loss is not None:
            total_loss = mlm_loss

        elif nsp_loss is not None:
            total_loss = nsp_loss

        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss,
            "nsp_loss": nsp_loss,
            "prediction_logits": prediction_logits,
            "nsp_logits": sequence_relationship_logits
        }

class BertForSequenceClassification(nn.Module):

    def __init__(
            self,
            config: BertConfig,
            num_classes: int
    ):
        super().__init__()

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            config.hidden_size,
            num_classes
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None
    ):
        outputs = self.bert(
            input_ids,
            token_type_ids,
            attention_mask
        )

        pooled_output = outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits,
                labels
            )

        return {
            "loss": loss,
            "logits": logits
        }

def create_mlm_input(
        input_ids: np.ndarray,
        tokenizer: Tokenizer,
        mlm_probability: float = 0.15,
        rng: Optional[np.random.Generator] = None
):
    if rng is None:
        rng = np.random.default_rng()

    input_ids = input_ids.copy()

    labels = np.full(
        input_ids.shape,
        -100,
        dtype=np.int64
    )

    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
    }

    candidate_indices = [
        i
        for i, token_id
        in enumerate(input_ids)
        if int(token_id)
        not in special_ids
    ]

    if not candidate_indices:
        return (
            input_ids,
            labels
        )

    selected_indices = [
        idx
        for idx in candidate_indices
        if rng.random()
        < mlm_probability
    ]

    if not selected_indices:
        selected_indices = [
            int(
                rng.choice(
                    candidate_indices
                )
            )
        ]

    for index in selected_indices:
        original_token = int(
            input_ids[index]
        )

        labels[index] = original_token
        probability = rng.random()

        if probability < 0.8:
            input_ids[index] = (
                tokenizer.mask_token_id
            )

        elif probability < 0.9:
            input_ids[index] = (
                rng.integers(
                    0,
                    tokenizer.vocab_size
                )
            )

        else:
            input_ids[index] = (
                original_token
            )

    return (
        input_ids,
        labels
    )

class BertPretrainingDataset(Dataset):

    def __init__(
            self,
            sentences: list[str],
            tokenizer: Tokenizer,
            max_length: int = 32,
            mlm_probability: float = 0.15,
            seed: int = 42
    ):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

        self.rng = np.random.default_rng(
            seed
        )

        self.examples = []

        self._build_examples()

    def _build_examples(self):
        n = len(
            self.sentences
        )

        for i in range(n - 1):
            self.examples.append(
                (
                    self.sentences[i],
                    self.sentences[i + 1],
                    1
                )
            )

            candidates = [
                j
                for j in range(n)
                if j != i + 1
                and j != i
            ]

            random_index = int(
                self.rng.choice(
                    candidates
                )
            )

            self.examples.append(
                (
                    self.sentences[i],
                    self.sentences[random_index],
                    0
                )
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,
            index
    ):
        text_a, text_b, nsp_label = (
            self.examples[index]
        )

        encoded = self.tokenizer.encode(
            text_a=text_a,
            text_b=text_b,
            max_length=self.max_length
        )

        input_ids = np.array(
            encoded["input_ids"],
            dtype=np.int64
        )

        masked_input_ids, mlm_labels = (
            create_mlm_input(
                input_ids,
                tokenizer=self.tokenizer,
                mlm_probability=self.mlm_probability,
                rng=self.rng
            )
        )

        return {
            "input_ids": torch.tensor(
                masked_input_ids,
                dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                encoded["token_type_ids"],
                dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                encoded["attention_mask"],
                dtype=torch.long
            ),
            "mlm_labels": torch.tensor(
                mlm_labels,
                dtype=torch.long
            ),
            "nsp_labels": torch.tensor(
                nsp_label,
                dtype=torch.long
            ),
        }


def train(
        model: BertForPreTraining,
        dataloader: DataLoader,
        device: torch.device,
        epochs: int = 10,
        learning_rate: float = 3e-4
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    model.to(
        device
    )

    for epoch in range(
        epochs
    ):
        model.train()

        total_loss = 0.0
        total_mlm_loss = 0.0
        total_nsp_loss = 0.0

        for batch in dataloader:
            input_ids = batch[
                "input_ids"
            ].to(device)

            token_type_ids = batch[
                "token_type_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            mlm_labels = batch[
                "mlm_labels"
            ].to(device)

            nsp_labels = batch[
                "nsp_labels"
            ].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                mlm_labels=mlm_labels,
                nsp_labels=nsp_labels
            )

            loss = outputs[
                "loss"
            ]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += (
                loss.item()
            )

            total_mlm_loss += (
                outputs[
                    "mlm_loss"
                ].item()
            )

            total_nsp_loss += (
                outputs[
                    "nsp_loss"
                ].item()
            )

        batches = len(
            dataloader
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{epochs} | "
            f"Loss: "
            f"{total_loss / batches:.4f} | "
            f"MLM: "
            f"{total_mlm_loss / batches:.4f} | "
            f"NSP: "
            f"{total_nsp_loss / batches:.4f}"
        )


@torch.no_grad()
def predict_mask(
        model: BertForPreTraining,
        tokenizer: Tokenizer,
        text: str,
        device: torch.device,
        max_length: int,
        top_k: int = 5
):
    model.eval()

    encoded = tokenizer.encode(
        text_a=text,
        max_length=max_length
    )

    input_ids = torch.tensor(
        [
            encoded[
                "input_ids"
            ]
        ],
        dtype=torch.long,
        device=device
    )

    token_type_ids = torch.tensor(
        [
            encoded[
                "token_type_ids"
            ]
        ],
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.tensor(
        [
            encoded[
                "attention_mask"
            ]
        ],
        dtype=torch.long,
        device=device
    )

    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask
    )

    logits = outputs[
        "prediction_logits"
    ]

    mask_positions = (
        input_ids[0]
        == tokenizer.mask_token_id
    ).nonzero(
        as_tuple=False
    ).flatten()

    if len(mask_positions) == 0:
        print(
            "В предложении нет [MASK]."
        )
        return

    for position in mask_positions:
        token_logits = logits[
            0,
            position
        ]

        probabilities = F.softmax(
            token_logits,
            dim=-1
        )

        top_probs, top_ids = torch.topk(
            probabilities,
            k=min(
                top_k,
                tokenizer.vocab_size
            )
        )

        print(
            "\n[MASK] predictions:"
        )

        for prob, token_id in zip(
            top_probs,
            top_ids
        ):
            token = tokenizer.id_to_token[
                int(token_id)
            ]

            print(
                f"{token:15s} "
                f"{float(prob):.4f}"
            )


@torch.no_grad()
def predict_next_sentence(
        model: BertForPreTraining,
        tokenizer: Tokenizer,
        sentence_a: str,
        sentence_b: str,
        device: torch.device,
        max_length: int
):
    model.eval()

    encoded = tokenizer.encode(
        text_a=sentence_a,
        text_b=sentence_b,
        max_length=max_length
    )

    input_ids = torch.tensor(
        [encoded["input_ids"]],
        dtype=torch.long,
        device=device
    )

    token_type_ids = torch.tensor(
        [encoded["token_type_ids"]],
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.tensor(
        [encoded["attention_mask"]],
        dtype=torch.long,
        device=device
    )

    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask
    )

    logits = outputs[
        "nsp_logits"
    ]

    probabilities = F.softmax(
        logits,
        dim=-1
    )

    not_next_probability = float(
        probabilities[
            0,
            0
        ]
    )

    is_next_probability = float(
        probabilities[
            0,
            1
        ]
    )

    print(
        "\nNSP:"
    )

    print(
        f"NotNext: "
        f"{not_next_probability:.4f}"
    )

    print(
        f"IsNext:  "
        f"{is_next_probability:.4f}"
    )

def count_parameters(
        model: nn.Module
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():
    sentences = [
        "машинное обучение позволяет находить закономерности в данных",
        "нейронные сети являются одним из методов машинного обучения",
        "трансформеры используют механизм внимания",
        "механизм внимания позволяет учитывать контекст слов",
        "bert является моделью на основе transformer encoder",
        "bert использует двунаправленное внимание",
        "модель анализирует левый и правый контекст слова",
        "masked language modeling используется для обучения bert",
        "некоторые токены заменяются специальным токеном mask",
        "модель должна восстановить исходные токены",
        "self attention использует query key и value",
        "attention вычисляет сходство между токенами",
        "multi head attention использует несколько голов внимания",
        "каждая голова может изучать различные зависимости",
        "feed forward network обрабатывает представление каждого токена",
        "residual connection помогает обучать глубокие нейронные сети",
        "layer normalization стабилизирует процесс обучения",
        "позиционные эмбеддинги содержат информацию о позиции токена",
        "segment embeddings позволяют различать два предложения",
        "токен cls используется как представление всей последовательности",
    ]

    tokenizer = Tokenizer()
    tokenizer.build_vocab(
        sentences
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size
    )

    max_length = 32

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=max_length,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    dataset = BertPretrainingDataset(
        sentences=sentences,
        tokenizer=tokenizer,
        max_length=max_length,
        mlm_probability=0.15
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )

    model = BertForPreTraining(
        config
    )

    print(
        "Trainable parameters:",
        f"{count_parameters(model):,}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    batch = next(
        iter(dataloader)
    )

    with torch.no_grad():
        test_outputs = model(
            input_ids=batch[
                "input_ids"
            ],
            token_type_ids=batch[
                "token_type_ids"
            ],
            attention_mask=batch[
                "attention_mask"
            ]
        )

    print(
        "\nMLM logits:",
        test_outputs[
            "prediction_logits"
        ].shape
    )

    print(
        "NSP logits:",
        test_outputs[
            "nsp_logits"
        ].shape
    )

    print(
        "\nTraining BERT...\n"
    )

    train(
        model=model,
        dataloader=dataloader,
        device=device,
        epochs=20,
        learning_rate=3e-4
    )

    predict_mask(
        model=model,
        tokenizer=tokenizer,
        text=(
            "bert использует "
            "[MASK] внимание"
        ),
        device=device,
        max_length=max_length,
        top_k=5
    )

    predict_next_sentence(
        model=model,
        tokenizer=tokenizer,
        sentence_a=(
            "трансформеры используют "
            "механизм внимания"
        ),
        sentence_b=(
            "механизм внимания позволяет "
            "учитывать контекст слов"
        ),
        device=device,
        max_length=max_length
    )


if __name__ == "__main__":
    main()
