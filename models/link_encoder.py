import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attrs import Attr

class Vanila_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Vanila_Attention, self).__init__()
        self.W = nn.Linear(hidden_size, 1)

    def forward(self, x):
        f = F.softmax(self.W(x), dim=-2)
        out = torch.sum(f * x, dim=-2)
        return out


class Link_Self_Att(nn.Module):
    def __init__(self, d_l):
        super(Link_Self_Att, self).__init__()
        self.temperature = d_l ** 0.5
        self.W_Q = nn.Linear(d_l, d_l, bias=False)
        self.W_K = nn.Linear(d_l, d_l, bias=False)
        self.W_V = nn.Linear(d_l, d_l, bias=False)

        self.link_layer_norm = nn.LayerNorm(d_l, eps=1e-6)

    def forward(self, H_hat_l, link_mask):
        Q = self.W_Q(H_hat_l)
        K = self.W_K(H_hat_l)
        V = self.W_V(H_hat_l)

        attn = torch.matmul(Q / self.temperature, K.transpose(-2, -1))
        attn = attn.masked_fill(~link_mask, -1e10)

        attention_score = F.softmax(attn, dim=-1)
        attention_score = torch.matmul(attention_score, V)

        att_out = self.link_layer_norm(attention_score + H_hat_l)
        return att_out


class Link_Encoder(nn.Module):
    def __init__(self, FLAGS):
        super(Link_Encoder, self).__init__()
        self.link_num = FLAGS.link_num
        self.batch_size = FLAGS.batch_size

        self.link_inp_dim = 256
        self.link_hidden_dim = 192
        self.cross_inp_dim = Attr.out_size("link")
        self.cross_hidden_dim = 64

        self.vanila_att = Vanila_Attention(256)

        self.rnn_link = nn.LSTM(self.link_inp_dim, self.link_hidden_dim, batch_first=True)
        self.rnn_cross = nn.LSTM(self.cross_inp_dim, self.cross_hidden_dim, batch_first=True)

        self.link_self_att = Link_Self_Att(d_l=256)

    def forward(self, route, seg_context_feat, cross):
        link_feat = self.vanila_att(seg_context_feat)

        link_lens = route["link_lens"]
        road_link_mask = torch.reshape(route["road_link_mask"], (self.batch_size, self.link_num)).bool().unsqueeze(1)

        link_lstm_enc = nn.utils.rnn.pack_padded_sequence(link_feat, link_lens.cpu(), batch_first=True,
                                                          enforce_sorted=False)
        link_lstm_enc, _ = self.rnn_link(link_lstm_enc)
        link_lstm_enc, _ = nn.utils.rnn.pad_packed_sequence(link_lstm_enc, batch_first=True)
        link_lstm_enc = F.pad(input=link_lstm_enc, pad=[0, 0, 0, self.link_num - link_lstm_enc.shape[1], 0, 0],
                              mode='constant', value=0)

        cross_lstm_enc = nn.utils.rnn.pack_padded_sequence(cross, link_lens.cpu(),
                                                           batch_first=True, enforce_sorted=False)
        cross_lstm_enc, _ = self.rnn_cross(cross_lstm_enc)
        cross_lstm_enc, _ = nn.utils.rnn.pad_packed_sequence(cross_lstm_enc, batch_first=True)
        cross_lstm_enc = F.pad(input=cross_lstm_enc, pad=[0, 0, 0, self.link_num - cross_lstm_enc.shape[1], 0, 0],
                               mode='constant', value=0)

        H_hat_l = torch.cat((link_lstm_enc, cross_lstm_enc), -1)
        link_context_feat = self.link_self_att(H_hat_l, road_link_mask)
        return link_context_feat
