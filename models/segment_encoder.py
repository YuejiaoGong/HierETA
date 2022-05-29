import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.attrs import Attr


class Seg_Self_Att(nn.Module):
    def __init__(self, d_s, seq_len, win_size, seg_tag=False):
        super(Seg_Self_Att, self).__init__()
        self.scaling = d_s ** 0.5
        self.W_Q = nn.Linear(d_s, d_s, bias=False)
        self.W_K = nn.Linear(d_s, d_s, bias=False)
        self.W_V = nn.Linear(d_s, d_s, bias=False)

        self.W_h = nn.Linear(d_s, 1, bias=True)
        self.W_g = nn.Linear(d_s, 1, bias=False)
        self.W_l = nn.Linear(d_s, 1, bias=False)

        self.feat_simi = self.get_mask(seq_len, win_size, seg_tag)
        self.seg_layer_norm = nn.LayerNorm(d_s, eps=1e-6)

    def forward(self, H_s, seg_mask):
        Q = self.W_Q(H_s)
        K = self.W_K(H_s)
        V = self.W_V(H_s)

        GP = torch.matmul(Q / self.scaling, K.transpose(-2, -1))
        GP = GP.masked_fill(~seg_mask, -1e10)
        G_Att = torch.matmul(F.softmax(GP, dim=-1), V)

        LP = GP.masked_fill(~self.feat_simi, -1e10)
        L_Att = torch.matmul(F.softmax(LP, dim=-1), V)

        gate = torch.sigmoid(self.W_h(H_s) + self.W_g(G_Att) + self.W_l(L_Att))
        Fusion = gate * L_Att + (1.0 - gate) * G_Att

        att_out = self.seg_layer_norm(Fusion + H_s)
        return att_out

    def get_mask(self, seq_len, win_size, seg=False):
        single_sided = math.floor(win_size / 2)
        mask = np.zeros((seq_len, seq_len))
        for i in range(-single_sided, single_sided + 1):
            mask += np.eye(seq_len, k=i)
        mask = torch.from_numpy(mask).cuda().bool().unsqueeze(0)
        if not seg:
            mask = mask.unsqueeze(0)
        return mask


class Segment_Encoder(nn.Module):
    def __init__(self, FLAGS):
        super(Segment_Encoder, self).__init__()

        self.emb_dim = Attr.out_size("seg") + Attr.out_size("ext")
        self.hidden_dim = 128
        self.link_num = FLAGS.link_num
        self.segment_num = FLAGS.segment_num
        self.batch_size = FLAGS.batch_size

        self.attr_mapping = nn.Linear(Attr.out_size("seg"), 128)
        self.relu = nn.ReLU()

        self.seg_lstm = nn.LSTM(self.emb_dim, self.hidden_dim, bidirectional=True, batch_first=True)
        self.linear_hidden = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_cell = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.seg_self_att = Seg_Self_Att(d_s=256, seq_len=50, win_size=FLAGS.win_size, seg_tag=True)

    def forward(self, route, ext, seg):
        ext = torch.unsqueeze(ext, dim=1)
        expand_ext = ext.expand(seg.size()[:2] + (ext.size()[-1],))
        segment_input = torch.cat((seg, expand_ext), dim=2)
        segment_input = torch.reshape(segment_input, (self.batch_size, self.link_num, self.segment_num, -1))

        link_seg_lens = route["link_seg_lens"]
        link_segment_mask = torch.reshape(route["road_segment_mask"],
                                          (self.batch_size, self.link_num, self.segment_num)).bool()

        emb_enc_inputs = []
        for i in range(segment_input.shape[1]):
            emb_enc_inputs.append(torch.squeeze(segment_input[:, i, :, :], dim=1))

        hidden = None
        seg_lstm_outs = []
        for i in range(len(emb_enc_inputs)):
            """cannot deal with variable-length series with length 0"""
            seq_lens_i = torch.clamp_min(link_seg_lens[:, i], min=1)
            enc_input = nn.utils.rnn.pack_padded_sequence(emb_enc_inputs[i], seq_lens_i.cpu(), batch_first=True,
                                                          enforce_sorted=False)
            enc_output, (hidden_h, hidden_c) = self.seg_lstm(enc_input, hidden)

            if i > 0:
                (hidden_h_pre, hidden_c_pre) = hidden
                real_seq_lens = (link_seg_lens[:, i] != 0).float().view(1, -1, 1)
                hidden_h = real_seq_lens * hidden_h + (1.0 - real_seq_lens) * hidden_h_pre
                hidden_c = real_seq_lens * hidden_h + (1.0 - real_seq_lens) * hidden_h_pre

            hidden = (hidden_h, hidden_c)
            enc_output, _ = nn.utils.rnn.pad_packed_sequence(enc_output, batch_first=True)
            enc_output = F.pad(input=enc_output, pad=[0, 0, 0, self.segment_num - enc_output.shape[1], 0, 0],
                               mode='constant', value=0)
            seg_lstm_outs.append(enc_output)

        seg_lstm_outs = torch.stack(seg_lstm_outs, dim=1)
        seg_context_feat = self.seg_self_att(seg_lstm_outs, link_segment_mask.unsqueeze(2))
        return seg_context_feat
