import torch
from torch import nn
import torch.nn.functional as F


class MultiLabelCrossEntropyLoss(nn.Module):
    def __init__(self, label_class_weights, is_soft_target=False):
        super().__init__()
        self._label_class_weights = label_class_weights
        self._is_soft_target = is_soft_target

    def forward(self, logits_list, true_labels):
        total_loss = 0.0

        for i, logits in enumerate(logits_list):
            class_weight = self._label_class_weights[i].to(logits.device)
            criterion = (
                SoftTargetCrossEntropy(class_weight)
                if self._is_soft_target
                else nn.CrossEntropyLoss(class_weight)
            )
            true_label = true_labels[i] if self._is_soft_target else true_labels[:, i]
            loss = criterion(logits, true_label)
            total_loss += loss

        return total_loss


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self._weights = weights

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)

        if self._weights is not None:
            self._weights = self._weights.to(x.device)
            weighted_log_probs = log_probs * self._weights.unsqueeze(0)
            loss = torch.sum(-target * weighted_log_probs, dim=-1)
        else:
            loss = torch.sum(-target * log_probs, dim=-1)

        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", ignore_index=-100):
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction
        self._ignore_index = ignore_index

    def forward(self, predictions, targets):
        if self._alpha is None:
            self._alpha = torch.ones(predictions.size(1), device=predictions.device)
        elif isinstance(self._alpha, (float, int)):
            self._alpha = torch.tensor(
                [self._alpha] * predictions.size(1), device=predictions.device
            )
        if predictions.dim() > 2:
            predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
            predictions = predictions.transpose(1, 2)
            predictions = predictions.contiguous().view(-1, predictions.size(2))
        targets = targets.view(-1, 1)

        logpt = F.log_softmax(predictions, dim=1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self._alpha.device != predictions.device:
            self._alpha = self._alpha.to(predictions.device)

        at = self._alpha.gather(0, targets.view(-1))
        logpt = logpt * at

        loss = -1 * (1 - pt) ** self._gamma * logpt
        if self._reduction == "mean":
            return loss.mean()
        elif self._reduction == "sum":
            return loss.sum()
        else:
            return loss
