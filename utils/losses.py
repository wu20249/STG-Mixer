import torch
import torch.nn.functional as F


def mse_with_regularizer_loss(inputs, targets, model, lamda=1.5e-3):
    lamda = 3e-3 #0.978
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2)
    reg_loss = lamda * reg_loss
    mse_loss = F.mse_loss(inputs, targets,  reduction='sum') / 2 #( / 47 * 8)
    loss = mse_loss + reg_loss
    return loss



