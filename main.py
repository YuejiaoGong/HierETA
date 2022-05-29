import os
import sys
import json
import argparse
import torch
import torch.optim as optim
from random import shuffle

import utils
import dataloading
from models import HierETA
from log import logger_tb, message_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--is_training', type=bool, default=True, help="training mode or not")
parser.add_argument('--segment_num', type=int, default=50, help="segment number per link")
parser.add_argument('--link_num', type=int, default=31, help="link number per route")
parser.add_argument('--win_size', type=int, default=3, help="window scale of neighboring segments")
parser.add_argument('--Lambda', type=float, default=0.4, help="weighting parameter in decoder")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--data_dir', type=str, default="./samples/", help="directory for route data-info storage")
parser.add_argument('--log_dir', type=str, default="logs")
parser.add_argument('--step_per_eval', type=int, default=100, help="training steps per evaluation")
parser.add_argument('--use_tb', type=bool, default=False, help='Use tensorboard to log training info')
parser.add_argument('--code_backup', type=bool, default=True, help='code backup or not')
parser.add_argument('--description', type=str, default="HierETA", help='description of current running experiments.')

FLAGS = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FLAGS.device = device
data_info = json.load(
    open('data-info/data_info.json',
         'r'))  # statistical information of different items, e.g., total distance of route, length and width of segment ...

# code backup and message logging
logger = logger_tb(FLAGS.log_dir, FLAGS.description, FLAGS.code_backup, FLAGS.use_tb)
sys.stdout = message_logger(logger.log_dir)


def train(model, optimizer):
    train_set = utils.get_train_files(FLAGS.data_dir)
    eval_set = utils.get_eval_files(FLAGS.data_dir)
    shuffle(train_set)
    print("train file nums: ", len(train_set))
    model.train()
    model.to(device)
    step = 0
    for epoch in range(FLAGS.epochs):
        print("train files " + str(train_set))
        print("eval files " + str(eval_set))
        print("--- Training epoch {} ---".format(epoch))

        for input_file in train_set:
            model.train()
            print('--- Train on file {} ---'.format(input_file))

            data_iter = dataloading.get_loader(input_file, FLAGS)
            data_iter_len = len(data_iter)
            for idx, attr in enumerate(data_iter):
                attr = utils.to_var(attr, device)

                optimizer.zero_grad()
                pred, label = model(attr)
                loss = utils.MAE(pred, label)
                loss.backward()
                optimizer.step()
                step += 1

                mape = utils.MAPE(pred, label)
                rmse = utils.RMSE(pred, label)
                if idx and (idx + 1) % 2 == 0:
                    print("--Progress: {:.4f} step:{} MAE_loss {:.4f} MAPE_loss {:.4f} RMSE_loss {:.4f}".format(
                        (idx + 1) * 100 / data_iter_len, step, loss, mape, rmse))

                if step and step % FLAGS.step_per_eval == 0:
                    with torch.no_grad():
                        weight_name = "model_epoch-{}_step-{}.pth".format(epoch, step)
                        print("Save weight file {}".format(weight_name))
                        check_point = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "step": step
                        }
                        torch.save(check_point, logger.log_dir + '/' + weight_name)
                        print(logger.log_dir + '/' + weight_name)
                        evaluate(model, eval_set)
                    model.train()

def evaluate(model, files):
    model.eval()
    MAE_loss = []
    MAPE_loss = []
    RMSE_loss = []

    for file_idx, input_file in enumerate(files):
        MAE_loss_single_file = []
        MAPE_loss_single_file = []
        RMSE_loss_single_file = []

        data_iter = dataloading.get_loader(input_file, FLAGS)
        for idx, attr in enumerate(data_iter):
            attr = utils.to_var(attr, device)
            pred, label = model(attr)

            mae = utils.MAE(pred, label)
            mape = utils.MAPE(pred, label)
            rmse = utils.RMSE(pred, label)

            MAE_loss_single_file.append(mae.item())
            RMSE_loss_single_file.append(rmse.item())
            MAPE_loss_single_file.append(mape.item())

            if idx > 10 and idx % 100 == 0:
                print("Evaluate Progress: {:.5f}".format((idx + 1) * 100 / len(data_iter)))
                print("step: {}, MAE_loss {:.5f}".format(idx, sum(MAE_loss_single_file) / len(MAE_loss_single_file)))
                print("step: {}, MAPE_loss {:.5f}".format(idx, sum(MAPE_loss_single_file) / len(MAPE_loss_single_file)))
                print("step: {}, RMSE_loss {:.5f}".format(idx, sum(RMSE_loss_single_file) / len(RMSE_loss_single_file)))
        print("***********************")
        print("Evaluate on file {}, MAE_loss {:.5f}".format(input_file,
                                                            sum(MAE_loss_single_file) / len(MAE_loss_single_file)))
        print("Evaluate on file {}, MAPE_loss {:.5f}".format(input_file,
                                                             sum(MAPE_loss_single_file) / len(MAPE_loss_single_file)))
        print("Evaluate on file {}, RMSE_loss {:.5f}".format(input_file,
                                                             sum(RMSE_loss_single_file) / len(RMSE_loss_single_file)))
        print("***********************\n\n")
        MAE_loss.extend(MAE_loss_single_file)
        MAPE_loss.extend(MAPE_loss_single_file)
        RMSE_loss.extend(RMSE_loss_single_file)
    MAPE = sum(MAPE_loss) / len(MAPE_loss)
    MAE = sum(MAE_loss) / len(MAE_loss)
    RMSE = sum(RMSE_loss) / len(RMSE_loss)
    print("\n--------final----------- \nMAPE: {:.5f} \nMAE:{:.5f} \nRMSE:{:.5f}\n".format(MAPE, MAE, RMSE))


def test(restore_epoch=0, restore_step=0, model_path=""):
    model = HierETA.HierETA_Net(FLAGS, data_info)
    file = "./{}/model_epoch-{}_step-{}.pth".format(model_path, restore_epoch, restore_step)
    check_point = torch.load(file)
    model.load_state_dict(check_point["model"])
    print("test model: " + file)
    model.to(device)
    evaluate(model, data_info["test_set"])


if __name__ == '__main__':

    model = HierETA.HierETA_Net(FLAGS, data_info)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=1e-5)

    if FLAGS.is_training:
        train(model, optimizer)
    else:
        model_path = "logs/2021-12-25-10-35-07_JustForDemo"
        files = os.listdir(model_path)
        files = list(file for file in files if file.endswith(".pth"))
        print(files)

        for pth in files:
            restore_epoch = int(pth.split("_")[1].split("-")[1])
            restore_step = int(pth.split("_")[2].split("-")[1].split(".")[0])
            test(restore_epoch, restore_step, model_path)
