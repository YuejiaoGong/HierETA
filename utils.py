import os
import json
import torch

data_info = json.load(open('data-info/data_info.json', 'r'))


def normalize(x, key, is_training=True):
    if is_training:
        mean = data_info["train_" + key + '_mean']
        std = data_info["train_" + key + '_std']
    else:
        mean = data_info["test_" + key + '_mean']
        std = data_info["test_" + key + '_std']
    return (x - mean) / std


def unnormalize(x, key, is_training):
    if is_training:
        mean = data_info["train_" + key + '_mean']
        std = data_info["train_" + key + '_std']
    else:
        mean = data_info["test_" + key + '_mean']
        std = data_info["test_" + key + '_std']
    return x * std + mean


def to_var(var, device):
    if torch.is_tensor(var):
        var = var.to(device)
        return var
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key], device)
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x, device), var)
        return var


def MAPE(pred, label):
    loss = torch.mean(torch.abs(pred - label) / label)
    return loss


def RMSE(pred, label):
    loss = torch.sqrt(torch.mean(torch.pow(pred - label, 2)))
    return loss


def MAE(pred, label):
    loss = torch.mean(torch.abs(pred - label))
    return loss


def get_train_files(dir_path):
    files = os.listdir(dir_path)
    train_files = [File for File in files if "train" in File]
    return train_files


def get_eval_files(dir_path):
    files = os.listdir(dir_path)
    eval_files = [File for File in files if "eval" in File]
    return eval_files


def get_test_files(dir_path):
    files = os.listdir(dir_path)
    test_files = [File for File in files if "test" in File]
    return test_files
