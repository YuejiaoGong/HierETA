import os
import json
import utils
import torch
import numpy as np
from random import shuffle
from torch.utils.data import Dataset, DataLoader

def get_network(link_attr_path):
    with open(link_attr_path, 'r') as f:
        road_network = json.load(f)
    return road_network

road_network = get_network("data-info/segment_attrs.json")  # Each item stores information about one road segment

class MySet(Dataset):
    def __init__(self, input_file, FLAGS):
        self.FLAGS = FLAGS
        self.root_dir = FLAGS.data_dir
        self.is_training = FLAGS.is_training
        self.content = open(os.path.join(self.root_dir, input_file), 'r').readlines()
        if self.is_training:
            shuffle(self.content)
        self.route_num = len(self.content)
        print("input_file: ", input_file, " route number: ", self.route_num)

    def __getitem__(self, idx):
        route = self.content[idx]
        route = eval(route.strip("\n").replace('"', "'"))

        gt_eta_time = int(route["gt_time"])

        timeID = route["timeID"]
        weekID = route["weekID"] - 1
        driverID = route["driverID"]

        segment_list = [item for sublist in route["segment_list_hier"] for item in sublist]  # list of all segments in a route
        segID, times, time_gaps, roadStates = self.get_attr(segment_list)
        lengths, segment_functional_levels, LaneNums, speedLimits, roadLevels, widths = self.get_segment_attr(segID)

        cross_info_id = []
        cross_info_time = []
        for j, cross in enumerate(route["cross_list"]):
            cross_info_id.append(cross[0])
            cross_info_time.append(cross[1])

        attr = {}

        attr["gt_eta_time"] = gt_eta_time

        attr["weekID"] = weekID
        attr["timeID"] = timeID
        attr["driverID"] = driverID

        attr["segID"] = segID
        attr["len"] = lengths
        attr["segment_functional_level"] = segment_functional_levels
        attr["laneNum"] = LaneNums
        attr["speedLimit"] = speedLimits
        attr["roadLevel"] = roadLevels
        attr["wid"] = widths
        attr["time_gap"] = time_gaps
        attr["time"] = times
        attr["roadState"] = roadStates

        attr["cross_info_id"] = cross_info_id
        attr["cross_info_time"] = cross_info_time

        attr["segment_list_hier"] = route["segment_list_hier"]  # nested lists, route -> links -> segments
        attr["cross_list"] = route["cross_list"]
        return attr

    def __len__(self):
        return self.route_num

    def get_attr(self, segment_list):
        segments = []
        index = []
        for i, x in enumerate(segment_list):
            segments.append(x[0])
            index.append(i)

        add_times = []
        total_time = 0
        add_times.append(total_time)

        for i in index[:-1]:
            total_time += segment_list[i][1]
            add_times.append(total_time)

        time_gaps = add_times
        times = [segment_list[i][1] for i in index]
        states = [segment_list[i][2] for i in index]
        return segments, times, time_gaps, states

    def get_segment_attr(self, segments):
        lengths = []
        road_levels = []
        speed_limits = []
        LaneNums = []
        segment_function_levels = []
        widths = []

        for i in segments:
            segment_attr = road_network[str(i)].strip("\n").split("\t")
            lengths.append(float(segment_attr[0]))
            segment_function_levels.append(int(segment_attr[1]) - 1)
            LaneNums.append(int(segment_attr[2]) - 1)
            speed_limits.append(float(segment_attr[3]))
            road_levels.append(int(segment_attr[4]) - 1)
            widths.append(int(segment_attr[5]))

        return lengths, segment_function_levels, LaneNums, speed_limits, road_levels, widths


def collate_fn(data, FLAGS):
    route_infos = ['gt_eta_time']
    ext_attrs = ['weekID', 'timeID', 'driverID']
    seg_attrs = ['segID', "roadState", "time_gap", 'time', 'segment_functional_level', 'laneNum', 'roadLevel',
                 'speedLimit',
                 'len', 'wid']

    cross_info_id = [item["cross_info_id"] for item in data]
    cross_info_time = [item["cross_info_time"] for item in data]
    for i in range(FLAGS.batch_size):
        while len(cross_info_id[i]) < FLAGS.link_num:
            cross_info_id[i].append(0)
            cross_info_time[i].append(0)
    cross_info_id = np.asarray(cross_info_id).reshape(FLAGS.batch_size, FLAGS.link_num)
    cross_info_time = np.asarray(cross_info_time).reshape(FLAGS.batch_size, FLAGS.link_num)

    link_lens = np.asarray([len(item['segment_list_hier']) for item in data])
    link_seg_lens = [[len(segment) for segment in item['segment_list_hier']] for item in data]
    for i in range(FLAGS.batch_size):
        while len(link_seg_lens[i]) < FLAGS.link_num:
            link_seg_lens[i].append(0)
    link_seg_lens = np.asarray(link_seg_lens).reshape(FLAGS.batch_size, FLAGS.link_num)

    link_seg_mask = link_seg_lens > 0
    link_seg_lens_cumsum = np.c_[np.zeros((FLAGS.batch_size, 1), dtype=np.int32), np.cumsum(link_seg_lens, axis=1)]

    attrs = {}
    lens = np.asarray([len(item['segID']) for item in data])
    attrs['segment_lens'] = torch.LongTensor(lens)

    for key in route_infos:
        x = torch.FloatTensor([item[key] for item in data])
        attrs[key] = utils.normalize(x, key, FLAGS.is_training)

    for key in ext_attrs:
        attrs[key] = torch.LongTensor([item[key] for item in data]) + 1
        if key == "key":
            x = torch.FloatTensor([item[key] for item in data])
            attrs[key] = utils.normalize(x, key, FLAGS.is_training)

    for key in seg_attrs:
        element = np.zeros((FLAGS.batch_size, FLAGS.link_num * FLAGS.segment_num), dtype=np.float32)
        seqs = [item[key] for item in data]
        for i in range(FLAGS.batch_size):
            seg_len = link_lens[i]
            for j in range(seg_len):
                ele = seqs[i][link_seg_lens_cumsum[i][j]:link_seg_lens_cumsum[i][j + 1]]
                element[i, j * FLAGS.segment_num:j * FLAGS.segment_num + len(ele)] = np.array(ele) + 1

        if key in ['time', 'len', 'wid', 'speed_lim', 'avg_speed']:
            padded = utils.normalize(element, key, FLAGS.is_training)
            padded = torch.from_numpy(padded).float()
        elif key == "time_gap":
            padded = torch.from_numpy(element).float()
        else:
            padded = torch.from_numpy(element).long()
        attrs[key] = padded

    segment_mask = element > 0
    attrs["link_lens"] = torch.from_numpy(link_lens).long()
    attrs["link_seg_lens"] = torch.from_numpy(link_seg_lens).long()

    attrs["road_segment_mask"] = torch.from_numpy(segment_mask.astype(float)).float()
    attrs["road_link_mask"] = torch.from_numpy(link_seg_mask.astype(float)).float()

    attrs["crossID"] = torch.from_numpy(cross_info_id).long()
    attrs["delayTime"] = torch.from_numpy(cross_info_time).float()
    return attrs


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.indices = list([i for i in range(self.count)])

    def __iter__(self):
        np.random.shuffle(self.indices)
        batches = (self.count - 1) // self.batch_size
        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


def get_loader(input_file, FLAGS):
    dataset = MySet(input_file=input_file, FLAGS=FLAGS)
    batch_sampler = BatchSampler(dataset, FLAGS.batch_size)
    data_loader = DataLoader(dataset=dataset, collate_fn=lambda x: collate_fn(x, FLAGS),
                             num_workers=0, batch_sampler=batch_sampler, pin_memory=True)
    return data_loader
