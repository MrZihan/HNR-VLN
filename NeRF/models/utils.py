import math
from turtle import heading
import torch
import numpy as np

def angle_feature(headings, device=None):
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    heading_enc = torch.zeros(len(headings), 64, dtype=torch.float32)

    for i, head in enumerate(headings):
        heading_enc[i] = torch.tensor(
                [math.sin(head), math.cos(head)] * (64 // 2))

    return heading_enc.to(device)

def dir_angle_feature(angle_list, device=None):
    feature_dim = 64
    batch_size = len(angle_list)
    max_leng = max([len(k) for k in angle_list]) + 1  # +1 for stop
    heading_enc = torch.zeros(
        batch_size, max_leng, feature_dim, dtype=torch.float32)

    for i in range(batch_size):
        for j, angle_rad in enumerate(angle_list[i]):
            heading_enc[i][j] = torch.tensor(
                [math.sin(angle_rad), 
                math.cos(angle_rad)] * (feature_dim // 2))

    return heading_enc


def angle_feature_with_ele(headings, device=None):
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    heading_enc = torch.zeros(len(headings), 128, dtype=torch.float32)

    for i, head in enumerate(headings):
        heading_enc[i] = torch.tensor(
            [
                math.sin(head), math.cos(head),
                math.sin(0.0), math.cos(0.0),  # elevation
            ] * (128 // 4))

    return heading_enc.to(device)

def angle_feature_torch(headings: torch.Tensor):
    return torch.stack(
        [
            torch.sin(headings),
            torch.cos(headings),
            torch.sin(torch.zeros_like(headings)),
            torch.cos(torch.zeros_like(headings))
        ]
    ).float().T

def dir_angle_feature_with_ele(angle_list, device=None):
    feature_dim = 128
    batch_size = len(angle_list)
    max_leng = max([len(k) for k in angle_list]) + 1  # +1 for stop
    heading_enc = torch.zeros(
        batch_size, max_leng, feature_dim, dtype=torch.float32)

    for i in range(batch_size):
        for j, angle_rad in enumerate(angle_list[i]):
            heading_enc[i][j] = torch.tensor(
            [
                math.sin(angle_rad), math.cos(angle_rad),
                math.sin(0.0), math.cos(0.0),  # elevation
            ] * (128 // 4))

    return heading_enc
    

def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask




def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    device = tensors[0].device
    output = torch.zeros(*size, dtype=dtype).to(device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def gen_seq_masks(seq_lens, max_len=None):
    if max_len is None:
        max_len = max(seq_lens)

    if isinstance(seq_lens, torch.Tensor):
        device = seq_lens.device
        masks = torch.arange(max_len).to(device).repeat(len(seq_lens), 1) < seq_lens.unsqueeze(1)
        return masks

    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=np.bool)
        
    seq_lens = np.array(seq_lens)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks

def get_angle_feature(heading, elevation=0., angle_feat_size=4):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_angle_fts(headings, elevations, angle_feat_size=4):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts

def get_point_angle_feature(base_heading=0., base_elevation=0., angle_feat_size=4):
    feature = np.empty((12, angle_feat_size), np.float32)
    for ix in range(12):
        heading = ix * math.radians(30)
        feature[ix, :] = get_angle_feature(heading, base_elevation, angle_feat_size)
    return feature

def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, z, y)
    dx = b[0] - a[0]
    dz = b[1] - a[1]
    dy = b[2] - a[2]
    if dx == dz == dy == 0:
        return 0, 0, 0

    xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    if b[2] < a[2]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz/xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation
    return heading, elevation, xyz_dist


def rectangular_to_polar(x, y):
    r = np.sqrt(np.square(x) + np.square(y))
    angle = np.arcsin(x/(r+1e-5))
    return angle


def calc_position_distance(a, b):
    # a, b: (x, z, y)
    dx = b[0] - a[0]
    dz = b[1] - a[1]
    dy = b[2] - a[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return dist

def pad_tensors_wgrad(tensors, lens=None):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    batch_size = len(tensors)
    hid = list(tensors[0].size()[1:])

    device = tensors[0].device
    dtype = tensors[0].dtype

    output = []
    for i in range(batch_size):
        if lens[i] < max_len:
            tmp = torch.cat(
                [tensors[i], torch.zeros([max_len-lens[i]]+hid, dtype=dtype).to(device)],
                dim=0
            )
        else:
            tmp = tensors[i]
        output.append(tmp)
    output = torch.stack(output, 0)
    return output
