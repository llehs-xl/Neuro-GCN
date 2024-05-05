import argparse
import os
from train import main_A
import torch

def main(args):
    config = {
        "repeat": 1,
        "TS":64,
        "lr":0.001,
        "batch_size":32,
        "s":50,
        "fold":5,
        "epoch":10000,
        "node":200,
        "num_class":1,
        "num_person":1,
        "g":8,
    }
    main_A(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_fea_dir', type=str,
                        default='/DATA/XL_XYK/xule_data/CamCAN_age_featureall_3_channels_5.npy')
    parser.add_argument('--all_label_dir', type=str,
                        default='/DATA/XL_XYK/xule_data/age.npy')

    # parser.add_argument('--save_dir', type=str, default='./modelsave/')
    # parser.add_argument('--train', type=bool, default=True)
    # parser.add_argument('--label_rate', type=float, default=0.2)

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main(args)