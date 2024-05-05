import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb
import torchvision
import torch.optim as optim
import random
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import pandas as pd
import math
from model import Model
from random import randint
from process import get_k_fold_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main_A (config, checkpoint_dir=None):

    corr_gust_list = []
    corr_gust_mark_list = []
    mse_list = []
    rmse_list = []
    mae_list = []
    featureall = np.load(config["all_fea_dir"])
    age = np.load(config["all_label_dir"])
    for lab in range(config["repeat"]):
        sample_num = len(age)
        sample_list = [i for i in range(len(age))]  # [0, 1, 2, 3]
        sample_list = random.sample(sample_list, sample_num)  # [1, 2]
        context2_2 = featureall[sample_list, :]  # array([[ 4,  5,  6,  7], [ 8,  9, 10, 11]])
        context1_1 = age[sample_list]  # array([2, 3])
        print(context2_2.shape)
        print(context1_1.shape)
        x = torch.Tensor(context2_2)
        y = torch.Tensor(context1_1)
        pre_age_dist = []
        rea_age_dist = []

        counter = []
        loss_history = []
        iteration_number = 0
        fin_pre = []
        fin_rea = []

        fin_channel_atten_tou = []
        fin_spatial_atten_tou = []

        fin_edg_imp = []

        TS = config["TS"]  # number of voters per test subject
        LR = config["lr"]   # learning rate
        batch_size = config["batch_size"]
        criterion = nn.MSELoss()
        training_loss = 0.0

        for window_size in [config["s"]]:
            W = window_size
            final_testing_accuracy = 0

            for fold in range(config["fold"]):
                print('-' * 80)
                print("Window Size {}, Fold {}".format(W, fold))
                print('-' * 80)
                best_test_acc = 1000000

                #             best_channel_atten_train=np.zeros((batch,64,1,1))
                #             best_spatial_atten_train=np.zeros((batch,1,50,200))
                #             best_channel_atten_test=np.zeros((batch,64,1,1))
                #             best_saptial_atten_test=np.zeros((batch,1,50,200))

                train_data, train_label, test_data, test_label = get_k_fold_data(config['fold'], fold, x, y)


                net = Model(training_data=train_data, num_class=config["num_class"], num_point=config["node"],
                            num_person=config["num_person"], groups=config["g"])
                net.to(device)
                optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)


                for epoch in range(config["epoch"]+1):
                    idx_batch = np.random.permutation(int(train_data.shape[0]))
                    idx_batch = idx_batch[:int(batch_size)]
                    train_data_batch = np.zeros((batch_size, 3, W, config["node"], 1))  # N,C,T,V,M
                    train_label_batch = train_label[idx_batch]

                    for i in range(batch_size):  # (32,1,256,200,1)
                        r1 = random.randint(0, train_data.shape[2] - W)
                        train_data_batch[i] = train_data[idx_batch[i], :, r1:r1 + W, :, :]

                    train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                    train_label_batch_dev = train_label_batch.to(device)
                    optimizer.zero_grad()
                    outputs, channel_atten_train_tou, spatial_atten_train_tou = net(train_data_batch_dev)
                    outputs = outputs.squeeze()
                    outputs = outputs.to(torch.float32)
                    train_label_batch_dev = train_label_batch_dev.to(torch.float32)
                    loss = criterion(outputs, train_label_batch_dev)
                    loss.backward()
                    optimizer.step()
                    iteration_number += 1
                    if iteration_number >= 200:
                        counter.append(iteration_number)
                        loss_history.append(loss.item())

                    if epoch % 1000 == 0:
                        idx_batch = np.random.permutation(int(test_data.shape[0]))
                        idx_batch = idx_batch[:int(batch_size)]
                        test_label_batch = test_label[idx_batch]

                        prediction = np.zeros((test_data.shape[0],))
                        voter = np.zeros((test_data.shape[0],))

                        channel_atten_test_all_tou = np.zeros((test_data.shape[0], 3, 1, 1))
                        spatial_atten_test_all_tou = np.zeros((test_data.shape[0], 1, config['s'], config['node']))

                        for v in range(TS):
                            idx = np.random.permutation(int(test_data.shape[0]))

                            batch_number = math.ceil(test_data.shape[0] / batch_size)
                            for k in range(batch_number):
                                if k == (batch_number - 1):
                                    idx_batch = idx[int(batch_size * k):int(test_data.shape[0])]
                                else:
                                    idx_batch = idx[int(batch_size * k):int(batch_size * (k + 1))]

                                test_data_batch = np.zeros((len(idx_batch), 3, W, config['node'], 1))

                                for i in range(len(idx_batch)):
                                    r1 = random.randint(0, test_data.shape[2] - W)
                                    test_data_batch[i] = test_data[idx_batch[i], :, r1:r1 + W, :, :]
                                test_data_batch_dev = torch.from_numpy(test_data_batch).float().to(device)
                                test_label_batch_dev = test_label_batch.to(device)
                                outputs, channel_atten_test_tou, spatial_atten_test_tou = net(test_data_batch_dev)

                                outputs = outputs.data.cpu().numpy()

                                channel_atten_test_tou = channel_atten_test_tou.data.cpu().numpy()
                                spatial_atten_test_tou = spatial_atten_test_tou.data.cpu().numpy()

                                #                             print("****************************************")
                                #                             print("outputs",outputs.shape)
                                #                             print("channel_atten_test",channel_atten_test.shape)
                                #                             print("spatial_atten_test",spatial_atten_test.shape)

                                #                             print("prediction",prediction.shape)
                                #                             print("channel_atten_test_all",channel_atten_test_all.shape)
                                #                             print("spatial_atten_test_all",spatial_atten_test_all.shape)
                                #                             print("idx_batch",idx_batch.shape)
                                for i in range(len(idx_batch)):
                                    which_person_number = idx_batch[i]

                                    channel_atten_test_all_tou[which_person_number] = channel_atten_test_all_tou[
                                                                                          which_person_number] + \
                                                                                      channel_atten_test_tou[i]
                                    spatial_atten_test_all_tou[which_person_number] = spatial_atten_test_all_tou[
                                                                                          which_person_number] + \
                                                                                      spatial_atten_test_tou[i]

                                prediction[idx_batch] = prediction[idx_batch] + outputs[:,0];
                                voter[idx_batch] = voter[idx_batch] + 1;

                        prediction = prediction / voter;

                        channel_atten_test_all_tou = channel_atten_test_all_tou / 64;
                        spatial_atten_test_all_tou = spatial_atten_test_all_tou / 64;

                        test_fold = pd.Series(prediction, dtype=np.float64)
                        real_fold = pd.Series(test_label, dtype=np.float64)
                        test_acc = round(test_fold.corr(real_fold), 4)
                        test_acc_mae = mean_absolute_error(real_fold, test_fold)
                        print("fold {},epoch {},test_acc {}, test_acc_mae {}\n".format(fold, epoch, test_acc,
                                                                                       test_acc_mae))
                        if test_acc_mae < best_test_acc:
                            best_test_acc = test_acc_mae
                            best_pre = prediction
                            best_rea = test_label

                            best_channel_atten_test_tou = channel_atten_test_all_tou  # (batch,64,1,1)
                            best_spatial_atten_test_tou = spatial_atten_test_all_tou

                        if epoch == config['epoch']:
                            fin_pre.extend(best_pre)
                            fin_rea.extend(best_rea)

                            fin_channel_atten_tou.extend(best_channel_atten_test_tou)
                            fin_spatial_atten_tou.extend(best_spatial_atten_test_tou)
                            print(len(fin_pre))
                            print(len(fin_rea))

                            print(len(fin_channel_atten_tou))
                            print(len(fin_spatial_atten_tou))
                    # torch.save(net.state_dict(),'checkpoint.pth')

                plt.plot(counter, loss_history)
                plt.show()


            pr2 = pd.Series(fin_pre, dtype=np.float64)
            re2 = pd.Series(fin_rea, dtype=np.float64)

            #         x = pd.Series(fin_pre)
            #         print(x)
            #         x.to_excel('/DATA/XL_XYK/xule_data/sandian/CamCAN_pre_age_no1{}.xlsx'.format(str(hh)),sheet_name='pre')
            #         y = pd.Series(fin_rea)
            #         print(y)
            #         y.to_excel('/DATA/XL_XYK/xule_data/sandian/CamCAN_rea_age_no1{}.xlsx'.format(str(hh)),sheet_name='rea')

            corr_gust = round(pr2.corr(re2), 4)
            corr_gust_mark = r2_score(re2, pr2)
            mae = mean_absolute_error(re2, pr2)
            mse = mean_squared_error(re2, pr2)
            rmse = sqrt(mse)
            print('corr', corr_gust)
            print('r2', corr_gust_mark)
            print('mae', mae)
            print('mse', mse)
            print('rmse', rmse)
            fin_edg_imp = np.array(fin_edg_imp)

            #         b=(fin_spatial_atten[0].reshape(50,200))
            #         ax = sns.heatmap(b)
            #         ax.set_xlabel('200 ROIs')
            #         ax.set_ylabel('200 ROIs')
            #         plt.show()
            #         figure = ax.get_figure()

            corr_gust_list.append(corr_gust)
            mae_list.append(mae)
            corr_gust_mark_list.append(corr_gust_mark)
            mse_list.append(mse)
            rmse_list.append(rmse)

