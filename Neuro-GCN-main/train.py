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
from process import get_k_fold_data,result_show,get_Acc,get_AUC
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
       
        x = torch.Tensor(featureall)
        y = torch.Tensor(age)
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
        
        all_fold_acc=[]
        all_fold_mae=[]

        TS = config["TS"]  # number of voters per test subject
        LR = config["lr"]   # learning rate
        batch_size = config["batch_size"]
        criterion =torch.nn.BCELoss()
        training_loss = 0.0

        for window_size in [config["s"]]:
            W = window_size
            final_testing_accuracy = 0

            for fold in range(config["fold"]):
                print('-' * 80)
                print("Window Size {}, Fold {}".format(W, fold))
                print('-' * 80)
              
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

                        acc = get_Acc(np.array(prediction), np.array(test_label))
                        AUC = get_AUC(np.array(test_label), np.array(prediction))
                        print("fold {}, epoch{},test_acc {}, test_AUC {}\n".format(fold,epoch,acc, AUC)) 
                         

                        if epoch == config['epoch']:
                            fin_pre.extend(prediction)
                            fin_rea.extend(test_label)
                            all_fold_acc.append(acc)
                            all_fold_mae.append(AUC)

                            fin_channel_atten_tou.extend(channel_atten_test_all_tou)
                            fin_spatial_atten_tou.extend(spatial_atten_test_all_tou)
                            print(len(fin_pre))
                            print(len(fin_rea))

                            print(len(fin_channel_atten_tou))
                            print(len(fin_spatial_atten_tou))
                    # torch.save(net.state_dict(),'checkpoint.pth')

                plt.plot(counter, loss_history)
                plt.show()

            
            
            print(f'--------------------------------------------------report results------------------------------------------------------')
            result_show(all_fold_acc,all_fold_mae)
            
            total_acc = get_Acc(np.array(fin_pre), np.array(fin_rea))
            total_AUC = get_AUC(np.array(fin_rea), np.array(fin_pre))
            print("total: acc {}, AUC {}\n".format(total_acc, total_AUC)) 
