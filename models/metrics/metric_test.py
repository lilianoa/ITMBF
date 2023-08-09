import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from bootstrap.lib.logger import Logger

class Metric_test(nn.Module):
    def __init__(self):
        super(Metric_test, self).__init__()

    def __call__(self, cri_out, net_out, batch):
        out = {}
        logits = net_out['logits'].data.cpu()
        _, pres = torch.max(logits, dim=1)
        label_id = batch['label_id'].data.cpu()
        acc, sen, spe, f1 = metric_test(pres, label_id)
        out['acc'] = torch.tensor(acc)
        out['sen'] = torch.tensor(sen)
        out['spe'] = torch.tensor(spe)
        out['f1'] = torch.tensor(f1)
        return out

def metric_test(output, target):
    cm = confusion_matrix(target, output)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    f1 = 2*tp / (2*tp + fp + fn)

    return acc*100, sen*100, spe*100, f1*100

