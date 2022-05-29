import torch
import torch.nn as nn

from models.attrs import Attr


class Seg_Att(nn.Module):
    def __init__(self):
        super(Seg_Att, self).__init__()
        self.w1 = nn.Linear(256, 256)
        self.w2 = nn.Linear(Attr.out_size("ext"), 256)
        self.v_seg = nn.Linear(256, 1, bias=False)
        self.softmax_seg = nn.Softmax(1)

    def forward(self, seg_context_feat, ext):
        e_seg = torch.sum(self.v_seg(torch.tanh(self.w1(seg_context_feat)) + self.w2(ext.unsqueeze(-2))), dim=(2))
        att_dist_seg = self.softmax_seg(e_seg)
        return att_dist_seg


class Link_Att(nn.Module):
    def __init__(self):
        super(Link_Att, self).__init__()
        self.w1 = nn.Linear(256, 256)
        self.w2 = nn.Linear(Attr.out_size("ext"), 256)
        self.v_link = nn.Linear(256, 1, bias=False)
        self.softmax_link = nn.Softmax(1)

    def forward(self, link_context_feat, ext):
        e_link = torch.sum(self.v_link(torch.tanh(self.w1(link_context_feat) + self.w2(ext.unsqueeze(-2)))), dim=2)
        att_dist_link = self.softmax_link(e_link)
        return att_dist_link


class Attention_Decoder(nn.Module):
    def __init__(self, FLAGS):
        super(Attention_Decoder, self).__init__()
        self.batch_size = FLAGS.batch_size
        self.link_num = FLAGS.link_num
        self.segment_num = FLAGS.segment_num
        self.Lambda = FLAGS.Lambda

        self.seg_level_att = Seg_Att()
        self.link_level_att = Link_Att()

        self.softmax = nn.Softmax(1)
        self.linear = nn.Linear(256, 1)

    def forward(self, route, seg_context_feat, link_context_feat, ext):
        seg_context_feat_reshaped = torch.reshape(seg_context_feat,
                                                  (self.batch_size, self.link_num * self.segment_num, -1))
        att_dist_seg = self.seg_level_att(seg_context_feat_reshaped, ext)
        att_dist_link = self.link_level_att(link_context_feat, ext)

        att_dist_guide = torch.mul(torch.reshape(att_dist_seg, (self.batch_size, self.link_num, self.segment_num)),
                                   att_dist_link.unsqueeze(dim=-1))

        segment_padding_mask = route["road_segment_mask"]

        masked_dist_seg = self.softmax(torch.reshape(
            att_dist_guide * segment_padding_mask.reshape(self.batch_size, self.link_num, self.segment_num),
            [self.batch_size, -1]))

        att_seg = torch.sum(
            torch.reshape(masked_dist_seg, [self.batch_size, self.link_num, self.segment_num, 1]) * seg_context_feat,
            [1, 2])

        att_link = torch.sum(torch.reshape(att_dist_link, [self.batch_size, self.link_num, 1]) * link_context_feat, [1])
        R = torch.reshape((1.0 - self.Lambda) * att_seg + self.Lambda * att_link, [self.batch_size, -1])
        output = self.linear(R)
        return output
