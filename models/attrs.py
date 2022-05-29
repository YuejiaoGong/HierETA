import torch
import torch.nn as nn

class Attr(nn.Module):
    ext_cates = [("weekID", 7 + 1, 3), ("timeID", 288 + 1, 5),
                 ("driverID", 200140 + 1, 16)]  # categorical attributes of external impact factors. plus one to avoid overflow.
    ext_conts = []

    seg_cates = [("segID", 1376566 + 1, 16), ("segment_functional_level", 8 + 1, 2), ("roadState", 5 + 1, 2),
                 ("laneNum", 6 + 1, 2), ("roadLevel", 8, 2)]
    seg_conts = ["wid", "speedLimit", "time", "len"]

    link_cates = [("crossID", 101008 + 1, 15)]
    link_conts = ["delayTime"]

    def __init__(self, FLAGS):
        super(Attr, self).__init__()
        self.batch_size = FLAGS.batch_size

        for name, dim_in, dim_out in Attr.ext_cates + Attr.seg_cates + Attr.link_cates:
            self.add_module("attr-" + name, nn.Embedding(dim_in, dim_out))

    def forward(self, attrs):
        ext = self.emb_helper(attrs, "ext")
        seg = self.emb_helper(attrs, "seg")
        link = self.emb_helper(attrs, "link")
        return ext, seg, link

    def emb_helper(self, attrs, type):
        Cates, Conts = Attr.type_helper(type)
        emb_list = []
        for name, dim_in, dim_out in Cates:
            embed = getattr(self, "attr-" + name)
            attr_t = attrs[name].view(self.batch_size, -1)
            attr_t = torch.squeeze(embed(attr_t))
            emb_list.append(attr_t)
        for name in Conts:
            attr_t = attrs[name].float()
            emb_list.append(attr_t.unsqueeze(-1))
        out = torch.cat(emb_list, -1)
        return out

    @staticmethod
    def out_size(type):
        Cates, Conts = Attr.type_helper(type)
        size = 0
        for name, dim_in, dim_out in Cates:
            size += dim_out
        size += len(Conts)
        return size

    @staticmethod
    def type_helper(types):
        if types == "ext":
            Cates = Attr.ext_cates
            Conts = Attr.ext_conts
        elif types == "seg":
            Cates = Attr.seg_cates
            Conts = Attr.seg_conts
        elif types == "link":
            Cates = Attr.link_cates
            Conts = Attr.link_conts
        else:
            raise Exception("must choose from ext, seg and link!")
        return Cates, Conts
