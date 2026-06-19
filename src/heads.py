import math

import torch
from torch import nn
from torch.nn import functional as F

from utils import PAD_LABEL_ID


def initialize_output_layer(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias, 0.01)


class SoftmaxHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)
        initialize_output_layer(self.classifier)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}

    def decode(self, logits: torch.Tensor, mask: torch.Tensor, id2label: dict[int, str]) -> list[list[str]]:
        pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
        masks = mask.detach().cpu().tolist()
        decoded = []
        for ids, row_mask in zip(pred_ids, masks):
            labels = [id2label[idx] for idx, keep in zip(ids, row_mask) if keep]
            decoded.append(labels)
        return decoded


class LinearChainCRF(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.start_transitions = nn.Parameter(torch.empty(num_labels))
        self.end_transitions = nn.Parameter(torch.empty(num_labels))
        self.transitions = nn.Parameter(torch.empty(num_labels, num_labels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def neg_log_likelihood(self, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        safe_labels = labels.masked_fill(~mask, 0)
        gold_score = self._compute_gold_score(emissions, safe_labels, mask)
        log_partition = self._compute_log_partition(emissions, mask)
        return (log_partition - gold_score).mean()

    def _compute_gold_score(self, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        batch_idx = torch.arange(batch_size, device=emissions.device)
        score = self.start_transitions[labels[:, 0]] + emissions[batch_idx, 0, labels[:, 0]]

        for t in range(1, seq_len):
            prev_labels = labels[:, t - 1]
            curr_labels = labels[:, t]
            transition_score = self.transitions[prev_labels, curr_labels]
            emission_score = emissions[batch_idx, t, curr_labels]
            score = score + (transition_score + emission_score) * mask[:, t]

        lengths = mask.long().sum(dim=1).clamp_min(1)
        last_labels = labels[batch_idx, lengths - 1]
        score = score + self.end_transitions[last_labels]
        return score

    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.start_transitions + emissions[:, 0]
        for t in range(1, emissions.size(1)):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        mask = mask.bool()
        paths = []
        for emission, row_mask in zip(emissions, mask):
            length = int(row_mask.long().sum().item())
            paths.append(self._viterbi_decode_one(emission[:length]))
        return paths

    def _viterbi_decode_one(self, emissions: torch.Tensor) -> list[int]:
        if emissions.size(0) == 0:
            return []
        score = self.start_transitions + emissions[0]
        history = []
        for t in range(1, emissions.size(0)):
            next_score = score.unsqueeze(1) + self.transitions
            best_score, best_path = next_score.max(dim=0)
            score = best_score + emissions[t]
            history.append(best_path)

        score = score + self.end_transitions
        best_last = int(score.argmax().item())
        best_path = [best_last]
        for backpointers in reversed(history):
            best_last = int(backpointers[best_last].item())
            best_path.append(best_last)
        best_path.reverse()
        return best_path


class CRFHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)
        initialize_output_layer(self.classifier)
        self.crf = LinearChainCRF(num_labels)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            if mask is None:
                mask = labels.ne(PAD_LABEL_ID)
            loss = self.crf.neg_log_likelihood(logits, labels, mask)
        return {"loss": loss, "logits": logits}

    def decode(self, logits: torch.Tensor, mask: torch.Tensor, id2label: dict[int, str]) -> list[list[str]]:
        pred_ids = self.crf.decode(logits, mask)
        return [[id2label[idx] for idx in row] for row in pred_ids]


class CascadePointerHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        entity_type_num: int,
        pointer_size: int = 64,
        max_span_len: int = 64,
    ):
        super().__init__()
        self.max_span_len = max_span_len
        self.start_classifier = nn.Linear(hidden_size, entity_type_num + 1)
        self.start_query = nn.Linear(hidden_size, pointer_size)
        self.end_key = nn.Linear(hidden_size, pointer_size)
        initialize_output_layer(self.start_classifier)
        initialize_output_layer(self.start_query)
        initialize_output_layer(self.end_key)
        self.start_loss = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)
        self.end_loss = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)

    def forward(
        self,
        features: torch.Tensor,
        labels: dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ):
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        start_logits = self.start_classifier(features)
        end_logits = self.compute_end_logits(features, mask)
        loss = None
        if labels is not None:
            loss = self.start_loss(
                start_logits.reshape(-1, start_logits.size(-1)),
                labels["start_labels"].reshape(-1),
            )
            if labels["end_labels"].ne(PAD_LABEL_ID).any():
                loss = loss + self.end_loss(
                    end_logits.reshape(-1, end_logits.size(-1)),
                    labels["end_labels"].reshape(-1),
                )
        return {"loss": loss, "logits": {"start": start_logits, "end": end_logits}}

    def compute_end_logits(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query = self.start_query(features)
        key = self.end_key(features)
        batch_size, seq_len, _ = query.shape
        invalid = torch.finfo(query.dtype).min
        scores = []
        for offset in range(self.max_span_len):
            if offset >= seq_len:
                scores.append(query.new_full((batch_size, seq_len), invalid))
                continue
            score = (query[:, : seq_len - offset] * key[:, offset:]).sum(dim=-1)
            score = score / math.sqrt(query.size(-1))
            valid = mask[:, : seq_len - offset] & mask[:, offset:]
            score = score.masked_fill(~valid, invalid)
            scores.append(F.pad(score, (0, offset), value=invalid))
        return torch.stack(scores, dim=-1)

    def decode(
        self,
        logits: dict[str, torch.Tensor],
        mask: torch.Tensor,
        id2entity: dict[int, str],
        texts: list[str],
    ) -> list[list[dict]]:
        start_ids = logits["start"].argmax(dim=-1).detach().cpu()
        end_offsets = logits["end"].argmax(dim=-1).detach().cpu()
        masks = mask.detach().cpu()
        decoded = []
        for row_starts, row_ends, row_mask, text in zip(start_ids, end_offsets, masks, texts):
            entities = []
            length = int(row_mask.long().sum().item())
            for start in range(length):
                start_id = int(row_starts[start].item())
                if start_id == 0:
                    continue
                end = start + int(row_ends[start].item()) + 1
                entities.append(
                    {
                        "text": text[start:end],
                        "type": id2entity[start_id - 1],
                        "start": start,
                        "end": end,
                    }
                )
            decoded.append(entities)
        return decoded


class EfficientGlobalPointerHead(nn.Module):
    def __init__(self, hidden_size: int, entity_type_num: int, head_size: int = 64):
        super().__init__()
        self.entity_type_num = entity_type_num
        self.head_size = head_size
        self.qk_proj = nn.Linear(hidden_size, head_size * 2)
        self.bias_proj = nn.Linear(hidden_size, entity_type_num * 2)
        initialize_output_layer(self.qk_proj)
        initialize_output_layer(self.bias_proj)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        logits = self.compute_logits(features)
        loss = None
        if labels is not None:
            if mask is None:
                mask = torch.ones(features.size()[:2], dtype=torch.bool, device=features.device)
            loss = self.multilabel_span_loss(logits, labels, mask)
        return {"loss": loss, "logits": logits}

    def compute_logits(self, features: torch.Tensor) -> torch.Tensor:
        qk = self.qk_proj(features)
        qw, kw = qk[..., : self.head_size], qk[..., self.head_size :]
        qw, kw = self.apply_rope(qw), self.apply_rope(kw)
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / (self.head_size**0.5)

        bias = self.bias_proj(features).view(features.size(0), features.size(1), self.entity_type_num, 2)
        start_bias = bias[..., 0].permute(0, 2, 1).unsqueeze(-1)
        end_bias = bias[..., 1].permute(0, 2, 1).unsqueeze(-2)
        return logits.unsqueeze(1) + start_bias + end_bias

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        position = torch.arange(seq_len, dtype=x.dtype, device=x.device).unsqueeze(1)
        indices = torch.arange(0, dim, 2, dtype=x.dtype, device=x.device)
        inv_freq = torch.pow(10000.0, -indices / dim)
        sinusoid = position * inv_freq
        sin = sinusoid.sin().repeat_interleave(2, dim=-1).unsqueeze(0)
        cos = sinusoid.cos().repeat_interleave(2, dim=-1).unsqueeze(0)
        x_even_odd = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).reshape(batch_size, seq_len, dim)
        return x * cos + x_even_odd * sin

    def multilabel_span_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        labels = labels.to(logits.dtype)
        valid = self.valid_span_mask(mask, logits.size(-1)).unsqueeze(1)
        logits = logits.masked_fill(~valid, -1e12)
        labels = labels * valid.to(labels.dtype)

        logits = logits.reshape(logits.size(0), logits.size(1), -1)
        labels = labels.reshape(labels.size(0), labels.size(1), -1)
        logits = (1.0 - 2.0 * labels) * logits
        logits_neg = logits - labels * 1e12
        logits_pos = logits - (1.0 - labels) * 1e12
        zeros = torch.zeros_like(logits[..., :1])
        neg_loss = torch.logsumexp(torch.cat([logits_neg, zeros], dim=-1), dim=-1)
        pos_loss = torch.logsumexp(torch.cat([logits_pos, zeros], dim=-1), dim=-1)
        return (neg_loss + pos_loss).mean()

    @staticmethod
    def valid_span_mask(mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        mask = mask.bool()
        span_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        triangle = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=mask.device))
        return span_mask & triangle.unsqueeze(0)

    def decode(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        id2entity: dict[int, str],
        texts: list[str],
        threshold: float = 0.0,
    ) -> list[list[dict]]:
        valid = self.valid_span_mask(mask, logits.size(-1)).unsqueeze(1)
        scores = logits.detach().masked_fill(~valid, float("-inf")).cpu()
        decoded = []
        for batch_idx, text in enumerate(texts):
            entities = []
            hits = (scores[batch_idx] > threshold).nonzero(as_tuple=False)
            for ent_id, start, end in hits.tolist():
                entities.append(
                    {
                        "text": text[start : end + 1],
                        "type": id2entity[ent_id],
                        "start": start,
                        "end": end + 1,
                    }
                )
            entities.sort(key=lambda x: (x["start"], x["end"], x["type"]))
            decoded.append(entities)
        return decoded
