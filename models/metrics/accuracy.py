import torch
import torch.nn as nn
from bootstrap.lib.logger import Logger

class Accuracy(nn.Module):

    def __init__(self):
        super(Accuracy, self).__init__()

    def __call__(self, cri_out, net_out, batch):
        out = {}
        logits = net_out['logits'].data.cpu()
        label_id = batch['label_id'].data.cpu()
        acc_out = accuracy(logits, label_id)
        out['accuracy'] = acc_out
        return out

def accuracy(output, target):
    batch_size = target.size(0)
    _, predictions = torch.max(output, dim=1)
    correct = predictions.eq(target.view_as(predictions))
    correct = correct.view(-1).float().sum(0)
    acc = correct.mul_(100.0 / batch_size)
    # Logger()('Overall Traditional Accuracy is {:.2f}'.format(acc))
    return acc