import torch
from torch import nn


class IDCNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        dropout: float = 0.3,
        dilations: list[int] | None = None,
        num_blocks: int = 1,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 1]
        self.num_blocks = num_blocks
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.proj = nn.Conv1d(embedding_dim, hidden_size, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.activation(self.proj(x))
        for _ in range(self.num_blocks):
            for layer in self.layers:
                x = self.activation(layer(x))
                x = self.dropout(x)
        return x.transpose(1, 2)


class IDCNNForTokenClassification(nn.Module):
    def __init__(self, encoder: IDCNNEncoder, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        features = self.encoder(input_ids)
        return self.head(features, labels, mask)
