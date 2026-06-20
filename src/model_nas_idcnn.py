import torch
from torch import nn


def build_search_operation(
    operation_type: str,
    input_channels: int,
    bottleneck_channels: int,
    kernel_size: int,
    dilation: int,
) -> tuple[nn.Module, int]:
    padding = (kernel_size // 2) * dilation
    if operation_type == "conv":
        return (
            nn.Conv1d(
                input_channels,
                bottleneck_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            bottleneck_channels,
        )
    if operation_type == "dwconv":
        return (
            nn.Conv1d(
                input_channels,
                input_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=input_channels,
            ),
            input_channels,
        )
    if operation_type == "sepconv":
        return (
            nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    input_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=input_channels,
                ),
                nn.Conv1d(input_channels, bottleneck_channels, kernel_size=1),
            ),
            bottleneck_channels,
        )
    if operation_type == "identity":
        return nn.Identity(), input_channels
    raise ValueError(f"unsupported NAS operation: {operation_type}")


class SearchCell(nn.Module):
    def __init__(self, channels: int, bottleneck_channels: int, operations: list[dict]):
        super().__init__()
        current_channels = channels
        modules = []
        activations = []
        self.operation_channels = []
        self.operation_specs = [dict(spec) for spec in operations]
        for spec in operations:
            operation, output_channels = build_search_operation(
                spec["type"],
                current_channels,
                bottleneck_channels,
                spec["kernel_size"],
                spec["dilation"],
            )
            modules.append(operation)
            activations.append(nn.Identity() if spec["type"] == "identity" else nn.ReLU())
            self.operation_channels.append((current_channels, output_channels))
            current_channels = output_channels
        self.operations = nn.ModuleList(modules)
        self.activations = nn.ModuleList(activations)
        self.final_conv = nn.Conv1d(current_channels, channels, kernel_size=3, padding=1)
        self.final_activation = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for operation, activation in zip(self.operations, self.activations):
            x = activation(operation(x))
            if mask is not None:
                x = x * mask
        x = self.final_activation(self.final_conv(x))
        if mask is not None:
            x = x * mask
        return x


class NASIDCNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        channels: int,
        ratio: float,
        cell_num: int,
        operations: list[dict],
        input_dropout: float = 0.35,
        hidden_dropout: float = 0.15,
    ):
        super().__init__()
        bottleneck_channels = int(channels * ratio)
        self.channels = channels
        self.ratio = ratio
        self.cell_num = cell_num
        self.embedding = nn.Embedding(vocab_size, channels, padding_idx=0)
        self.input_dropout = nn.Dropout(input_dropout)
        self.initial_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.initial_activation = nn.ReLU()
        self.cell = SearchCell(channels, bottleneck_channels, operations)
        self.hidden_dropout = nn.Dropout(hidden_dropout)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        token_mask = mask.unsqueeze(-1).to(torch.float) if mask is not None else None
        x = self.embedding(input_ids)
        if token_mask is not None:
            x = x * token_mask
        x = self.input_dropout(x).transpose(1, 2)
        conv_mask = token_mask.transpose(1, 2) if token_mask is not None else None
        x = self.initial_activation(self.initial_conv(x))
        if conv_mask is not None:
            x = x * conv_mask
        for _ in range(self.cell_num):
            x = self.cell(x, conv_mask)
        features = self.hidden_dropout(x.transpose(1, 2))
        if token_mask is not None:
            features = features * token_mask
        return features


class NASIDCNNForNER(nn.Module):
    def __init__(self, encoder: NASIDCNNEncoder, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, input_ids: torch.Tensor, labels=None, mask: torch.Tensor | None = None):
        features = self.encoder(input_ids, mask)
        return self.head(features, labels, mask)
