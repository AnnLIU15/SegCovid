# @Author: ZhaoYang
# @Date:   2021-04-23 18:58:51
# @Last Modified by:   ZhaoYang
# @Last Modified time: 2021-05-26 20:26:31
import argparse


def getConfig(stage):
    parser_ = argparse.ArgumentParser(
        description='训练的参数')
    
    parser_.add_argument("--device", type=str, default='cuda')
    parser_.add_argument("--num_classes", type=int, default=3,
                         help="This refers to the number of classes in the segmentation mode, so either 2 or 3")
    parser_.add_argument("--model_name", type=str, default='UNet')

    if stage == "train":
        parser_.add_argument("--batch_size", type=int,
                             default=1, help='batch_size')
        parser_.add_argument("--start_epoch", type=str, default=1)
        parser_.add_argument("--num_epochs", type=int, default=100)
        parser_.add_argument("--preTrainedSegModel", type=str, default=None,
                             help="预训练的模型")
        parser_.add_argument("--save_dir", type=str, default="./output/saved_models",
                             help="Directory to save checkpoints")
        parser_.add_argument("--train_data_dir", type=str, default='./data/seg/train',
                             help="Path to the training data. Must contain images and binary masks")
        parser_.add_argument("--val_data_dir", type=str, default='./data/seg/test',
                             help="Path to the validation data. Must contain images and binary masks")
        # parser_.add_argument("--batch_size", type=int, default=8, help="Implemented only for batch size = 1")
        parser_.add_argument("--save_every", type=int, default=5)
        parser_.add_argument("--lrate", type=float,
                             default=1e-3, help="initial Learning rate")
        parser_.add_argument('--log_name', type=str,
                             default=None, help='中断后继续训练记载')
        parser_.add_argument('--weight', type=list,
                             default=[1, 20, 20], help='交叉熵权值')
    elif stage == "test":
        parser_.add_argument("--pth", type=str, default='./output/saved_models/best_epoch_model.pth',
                             help="训练好的pth路径，模型必须包含以下参数"
                                  "model_weights, optimizer_state, anchor_generator")
        parser_.add_argument("--test_data_dir", type=str, default='./data/seg/test/',
                             help="Path to the test data. Must contain images and may contain binary masks")
        parser_.add_argument("--save_seg", type=str,
                             default='./output/segResult/')
    model_args = parser_.parse_args()
    return model_args
