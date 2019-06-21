import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, loss_weights=0.8, delta1=0.1, delta2=10):
        super(Loss, self).__init__()
        self.loss_weights = [loss_weights ** (3-i) for i in range(4)]
        self.delta1 = nn.Parameter(torch.Tensor([delta1]))
        self.delta2 = nn.Parameter(torch.Tensor([delta2]))

    def forward(self, outputs, depths, labels):
        # loss = self.delta1 + self.delta2
        loss = 0
        for i, pair in enumerate(outputs):
            # loss = loss + torch.exp(-self.delta1) * self.loss_weights[i] * sim_depth_loss(depths, pair[0]) + \
            #        torch.exp(-self.delta2) * self.loss_weights[i] * cross_entropy2d(pair[1], labels)

            loss = loss + self.delta1 * self.loss_weights[i] * sim_depth_loss(depths, pair[0]) + \
                   self.delta2 * self.loss_weights[i] * cross_entropy2d(pair[1], labels)
        return loss


def sim_depth_loss(y_true, y_pred):
    mask = (y_true > 0).float()
    y_pred = mask * y_pred
    c = 0.2 * torch.max(torch.abs(y_pred-y_true))
    loss = torch.abs(y_true-y_pred) * (torch.abs(y_pred - y_true) <= c).float() + (torch.pow(y_true-y_pred, 2) + c**2)/(2*c) * (torch.abs(y_pred-y_true) > c).float()

    loss = torch.mean(loss)

    return loss


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)



