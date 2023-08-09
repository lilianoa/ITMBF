# ref: https://github.com/zcyang/imageqa-san/blob/master/src/san_att_lstm_twolayer_theano.py
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Attention(nn.Module):
    def __init__(self, d=1024, k=512, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        # N * 196 * 1024 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq).unsqueeze(dim=1)
        # N * 196 * 512
        ha = F.tanh(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha, dim=1)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u


class SANModel(nn.Module):
    def __init__(self,
            w_emb,
            t_emb,
            img_enc,
            classif,
            attention={}):
        super(SANModel, self).__init__()
        self.w_emb = w_emb
        self.t_emb = t_emb
        self.img_enc = img_enc

        emb_size = attention['input_dims'][0]
        i_emb_size = attention['input_dims'][1]
        att_ff_size = attention['att_ff_size']
        output_size = attention['output_dim']
        num_att_layers = attention['num_att_layers']
        self.linear = nn.Linear(i_emb_size, emb_size)
        self.san = nn.ModuleList(
            [Attention(d=emb_size, k=att_ff_size)] * num_att_layers)

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(emb_size, output_size))
        self.classif = classif

    def forward(self, batch):
        v = batch['img']
        t = batch['caption']

        v_emb = self.img_enc(v)   # v:[batch, 2048, 7, 7]
        batch_size = v_emb.size(0)
        width = v_emb.size(2)
        high = v_emb.size(3)
        v_emb = v_emb.view(batch_size, -1, width*high).permute(0,2,1)   # v: [batch, width*high, dim_v]
        v_emb = self.linear(v_emb)  # v: [batch, width*high, num_him]
        w_emb = self.w_emb(t)
        t_emb = self.t_emb.forward_all(w_emb)  # w_emb: [batch, seq_length]-->t_emb:[batch, seq_length, num_him]
        t_emb = t_emb.sum(dim=1)  # [batch, num_him]
        for att_layer in self.san:
            u = att_layer(v_emb, t_emb)
        z = self.mlp(u)
        logits = self.classif(z)
        out = {'logits': logits}
        return out