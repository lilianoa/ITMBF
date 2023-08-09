"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is modified from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from modules.fc import FCNet
from modules.bc import BCNet


class BanModel(nn.Module):
    def __init__(self,
                 w_emb,
                 t_emb,
                 img_enc,
                 classif,
                 attention={}
                 ):
        super(BanModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = t_emb
        input_dim = attention['input_dims'][1]
        num_hid = attention['input_dims'][0]
        self.glimpse = attention['gamma']
        self.v_att = BiAttention(input_dim, num_hid, num_hid, self.glimpse)
        self.img_enc = img_enc
        b_net = []
        q_prj = []
        for i in range(self.glimpse):
            b_net.append(BCNet(input_dim, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.classif = classif

    def forward(self, batch):
        """Forward

        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]
        labels: [batch]

        return: logits, not probs
        """
        v = batch['img']
        q = batch['caption']
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        v_emb = self.img_enc(v) # v:[batch, 2048, 7, 7]
        batch_size = v_emb.size(0)
        width = v_emb.size(2)
        high = v_emb.size(3)
        v_emb = v_emb.view(batch_size, -1, width*high).permute(0,2,1)   # v: [batch, width*high, dim_v]

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v_emb, q_emb)   # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att[:,g,:,:]) # b x l x h
            atten, _ = logits[:,g,:,:].max(2)
            # embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            # q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classif(q_emb.sum(1))
        out = {'logits': logits}

        return out

class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True, logit=False, mask_with=-float('inf')):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, mask_with)

        if not logit:
            p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
            return p.view(-1, self.glimpse, v_num, q_num), logits

        return logits
