import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.fusions.factory import factory as factory_fusion

class ITMBFNet(nn.Module):
# 包括文本编码器（是否使用自注意力）、图像编码器、多模态双线性注意力模块
# 多模态双线性注意力模块：输入文本特征和图像特征，输出多模态表征
    def __init__(self,
            w_emb,
            t_emb,
            img_enc,
            classif,
            attention={},):
        super(ITMBFNet, self).__init__()
        # Modules
        self.w_emb = w_emb
        self.t_emb = t_emb
        self.img_enc = img_enc
        # self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        self.attention = Attention(attention)
        self.classif = classif

    def forward(self, batch):
        v = batch['img']
        t = batch['caption']

        v_emb = self.img_enc(v)   # v:[batch, 2048, 7, 7]
        w_emb = self.w_emb(t)
        t_emb = self.t_emb.forward_all(w_emb)  # w_emb: [batch, seq_length]-->t_emb:[batch, seq_length, num_him]
        t_emb = t_emb.sum(dim=1)  # [batch, num_him]
        z = self.attention(t_emb, v_emb)
        logits = self.classif(z)
        out = {'logits': logits}
        return out


class Attention(nn.Module):
# 将完整的多模态双线性注意力模块写在这个部分
# 包括第一个双线性函数：产生v，第二个双线性函数：产生a，a进行归一化和softmax，最后av得到score
# 注意：第一个双线性函数输出向量，第二个双线性函数输出标量
# 多头注意力n_glimpses
# Attention的超参数有三个：第一个双线性函数，第二个双线性函数，glimpses的数量
# 输入是两个模态的输入：图像特征（maps)和文本特征（向量），输出是多模态表征（向量）

    def __init__(self, fusions={}):
        super(Attention, self).__init__()
        # self.nb_heads = fusions['nb_heads']
        self.v_fusion = factory_fusion(fusions['v_fusion'])
        self.a_fusion = factory_fusion(fusions['a_fusion'])

    def forward(self, t, v):
        '''

        :param t: [batch, dim_t]  # 取最后token作为文本representation
        :param v: [batch, dim_v, w, h]
        :return: z: [batch, dim_z]
        '''
        # 由于fusion输入是两个向量
        # 而多模态双线性注意力模块的输入是图像maps和文本特征向量
        # 因此需要对图像maps进行处理才能输入到fusion中
        # 或者对fusion进行修改，使得输入是矩阵
        # 如果我们计算maps每个分量和vector的双线性，然后拼回来，和我们计算整个maps和vector的双线性是一样的话
        # 那么我们可以这样做，如果不一样就不可以，但是目前我还没办法证明是否相等
        # 我们只能先根据这么定义（通过计算矩阵分量，即向量，的双线性来得到矩阵的双线性）去计算
        batch_size = v.size(0)
        width = v.size(2)
        high = v.size(3)
        v = v.view(batch_size, -1, width*high).permute(0,2,1)   # v: [batch, width*high, dim_v]
        alpha = self.process_attention(t, v)    # t: [batch, dim_t]  alpha:[batch, wh, 1]
        values = self.process_fusion(t, v)      # values: [batch, wh, dim_z]  if nb_heads, values: [batch*nb_heads, wh, dim_z/nb_heads]

        alpha = F.softmax(alpha, dim=1)

        alpha = alpha.expand_as(values)
        z_out = alpha * values   # z_out: [batch, wh, dim_z] if nb_heads, values: [batch*nb_heads, wh, dim_z/nb_heads]
        # z_out = self.transpose_output(z_out, self.nb_heads)  # z_out: [batch, wh, dim_z] if nb_heads, values: [batch, wh, dim_z/nb_heads*nb_heads]
        z_out = z_out.sum(1)
        return z_out  # [batch, dim_z]


    def process_attention(self, t, v):
        batch_size = t.size(0)
        n_regions = v.size(1)
        t = t[:, None, :].expand(t.size(0), n_regions, t.size(1))
        alpha = self.a_fusion([
            t.contiguous().view(batch_size*n_regions, -1),
            v.contiguous().view(batch_size*n_regions, -1)
        ])
        alpha = alpha.view(batch_size, n_regions, -1)
        # 多头注意力
        # alpha = self.transpose_qkv(alpha, self.nb_heads)
        return alpha


    def process_fusion(self, t, v):
        batch_size = t.size(0)
        n_regions = v.size(1)
        t = t[:, None, :].expand(t.size(0), n_regions, t.size(1))
        values = self.v_fusion([
            t.contiguous().view(batch_size * n_regions, -1),
            v.contiguous().view(batch_size * n_regions, -1)
        ])
        values = values.view(batch_size, n_regions, -1)
        # 多头注意力
        # values = self.transpose_qkv(values, self.nb_heads)
        return values

    def transpose_qkv(self, x, nb_heads):
        '''为了多头注意力的并行计算而变换形状
        :param x: [batch_size, n_regions, num_hiddens]
        :param nb_heads: 注意力头的个数
        输出x: [batch_size, n_regions, num_heads, num_hiddens/num_heads]会改变维数，所以变成
        [batch_size*num_heads, n_regions, num_hiddens/num_heads]
        :return: x: [batch_size*num_heads, n_regions, num_hiddens/num_heads]
        '''
        x = x.reshape(x.size(0), x.size(1), nb_heads, -1)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(-1, x.size(2), x.size(3))

    def transpose_output(self, x, nb_heads):
        '''逆转transpose_qkv函数的操作'''
        x = x.reshape(-1, nb_heads, x.size(1), x.size(2))
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.size(0), x.size(1), -1)


