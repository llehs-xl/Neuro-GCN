import numpy as np
import torch


def compute_A(time_series3):
    time_series1=time_series3.squeeze()
    time_series1=time_series1[:,0,:,:]
    c=time_series1[0]
    for i in range(1,len(time_series3)):
        c=np.r_[c,time_series1[i]]
    this_A=np.corrcoef(c.T)
    return this_A


def get_k_fold_data(k,i,x,y):
    assert k>1
    fold_size=x.shape[0]//k
    x_train,y_train=None,None
    for j in range(k):
        idx=slice(j*fold_size,(j+1)*fold_size)
        x_part,y_part=x[idx,:],y[idx]
        if j==i:
            x_valid,y_valid=x_part,y_part
        elif x_train is None:
            x_train,y_train=x_part,y_part
        else:
            x_train=torch.cat((x_train,x_part),dim=0)
            y_train=torch.cat((y_train,y_part),dim=0)
    return x_train,y_train,x_valid,y_valid


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod




from scipy.stats import pearsonr
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score
def avg_std(Y_data):
    avg = np.mean(Y_data)
    std = np.std(Y_data)
    return avg, std

def result_show(all_fold_acc,all_fold_mae):
    cor_mean, cor_std = avg_std(all_fold_acc)
    mae_mean, mae_std = avg_std(all_fold_mae)
    print("average of 5-fold: acc=",round(cor_mean,3),"+-",round(cor_std,3), "auc=",round(mae_mean,3),"+-",round(mae_std,3))
    print(f"acc_list={all_fold_acc} auc_list={all_fold_mae}")
    
    
    
def get_Acc(predata,data_label): 
    predata=np.squeeze(predata)
    data_label=np.squeeze(data_label)

    predata = np.where(predata > 0.5, 1, 0)
    accuracy = np.where(predata == data_label,1,0).mean()

    return accuracy

def get_AUC(test_targets,output):
    test_targets=np.squeeze(test_targets)
    output=np.squeeze(output)
    A=np.sort(output)
    I=np.argsort(output)
    M=0
    N=0
    for i in range(len(output)):
        if (test_targets[i]==1):
            M=M+1
        else:
            N=N+1
    sigma=0
    for i in range(M+N):
        if test_targets[I[i]]==1:
            sigma=sigma+i

    result = (sigma - (M + 1) * M / 2) / (M * N)
    return result
