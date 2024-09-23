import torch


class AvgClassAccuracy:
    def __init__(self, num_labels_list):
        self._num_labels_list = num_labels_list

        self._num_attr = len(self._num_labels_list)

    def __call__(self, logits_list, gt_labels):
        gt_labels = gt_labels.detach()
        pred_labels = torch.cat(
            [logits.detach().argmax(-1, keepdim=True) for logits in logits_list], dim=1
        )

        per_class_acc = []
        for attr_idx in range(self._num_attr):
            for idx in range(self._num_labels_list[attr_idx]):
                target = gt_labels[:, attr_idx]
                pred = pred_labels[:, attr_idx]
                correct = torch.sum((target == pred) * (target == idx))
                total = torch.sum(target == idx)

                if total.item() > 0:
                    acc = float(correct.item()) / float(total.item())

                else:
                    acc = 0.0

                per_class_acc.append(acc)

        return sum(per_class_acc) / len(per_class_acc)
