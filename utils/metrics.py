import torch


def r2(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    def _to_tensor_nograd(x, device=None, dtype=torch.float32):
        if torch.is_tensor(x):
            return x.detach().to(device=device, dtype=dtype)
        else:
            return torch.as_tensor(x, device=device, dtype=dtype)

    y = _to_tensor_nograd(y, device=pred.device if torch.is_tensor(pred) else None)
    pred = _to_tensor_nograd(pred, device=y.device)
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)

