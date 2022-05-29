import torch.nn as nn

from models.attrs import Attr
from models.segment_encoder import Segment_Encoder
from models.link_encoder import Link_Encoder
from models.decoder import Attention_Decoder


class HierETA_Net(nn.Module):
    def __init__(self, FLAGS, data_info):
        super(HierETA_Net, self).__init__()
        self.batch_size = FLAGS.batch_size
        self.data_info = data_info

        """Attribute Feature Extractor"""
        self.attr_net = Attr(FLAGS)

        """Hierarchical Self-Attention Network"""
        self.seg_enc = Segment_Encoder(FLAGS)
        self.link_enc = Link_Encoder(FLAGS)

        """Hierarchy-Aware Attention Decoder"""
        self.decoder = Attention_Decoder(FLAGS)

        self.init_weight()

    def init_weight(self):
        for name, params in self.named_parameters():
            if "norm" in name.lower():
                continue
            if name.find('.bias') != -1:
                params.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform_(params.data)

    def forward(self, route):
        ext, seg, cross = self.attr_net(route)

        seg_context_feat = self.seg_enc(route, ext, seg)
        link_context_feat = self.link_enc(route, seg_context_feat, cross)

        pred = self.decoder(route, seg_context_feat, link_context_feat, ext)

        time_mean = self.data_info['train_gt_eta_time_mean']
        time_std = self.data_info['train_gt_eta_time_std']
        label = route['gt_eta_time'].view(self.batch_size, 1)
        label = label * time_std + time_mean
        pred = pred * time_std + time_mean
        return pred, label
