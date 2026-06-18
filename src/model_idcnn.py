import torch
from torch import nn


class IDCNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 300,
        input_dropout: float = 0.35,
        hidden_dropout: float = 0.15,
        dilations: list[int] | None = None,
        kernel_size: int = 3,
        num_blocks: int = 1,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 1]
        self.num_blocks = num_blocks
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.initial_conv = nn.Conv1d(
            embedding_dim,
            hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            dilation=1,
        )
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=kernel_size,
                    padding=(kernel_size // 2) * dilation,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.activation = nn.ReLU()
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.initial_conv.weight)
        nn.init.constant_(self.initial_conv.bias, 0.01)
        for layer in self.layers:
            nn.init.dirac_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_all_blocks: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        x = self.embedding(input_ids)
        token_mask = mask.unsqueeze(-1).to(dtype=x.dtype) if mask is not None else None
        if token_mask is not None:
            x = x * token_mask
        x = self.input_dropout(x)
        x = x.transpose(1, 2)
        conv_mask = token_mask.transpose(1, 2) if token_mask is not None else None
        x = self.activation(self.initial_conv(x))
        if conv_mask is not None:
            x = x * conv_mask
        block_features = []
        for _ in range(self.num_blocks):
            for layer in self.layers:
                x = self.activation(layer(x))
                if conv_mask is not None:
                    x = x * conv_mask
            features = self.hidden_dropout(x.transpose(1, 2))
            if token_mask is not None:
                features = features * token_mask
            block_features.append(features)
        return block_features if return_all_blocks else block_features[-1]


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
        if labels is None:
            features = self.encoder(input_ids, mask)
            return self.head(features, None, mask)

        block_features = self.encoder(input_ids, mask, return_all_blocks=True)
        block_outputs = [self.head(features, labels, mask) for features in block_features]
        final_output = block_outputs[-1]
        final_output["loss"] = torch.stack([output["loss"] for output in block_outputs]).sum()
        return final_output
