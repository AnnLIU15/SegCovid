# @Author: ZhaoYang
# @Date:   2021-04-23 18:58:51
# @Last Modified by:   ZhaoYang
# @Last Modified time: 2021-06-13 15:41:16
import argparse


def getConfig(stage):
    parser_ = argparse.ArgumentParser(
        description='训练或的参数')
    parser_.add_argument("--pth", type=str, default=None,
                         help="训练好的pth路径，模型必须包含以下参数"
                         "model_weights, optimizer_state")
    parser_.add_argument("--device", type=str, default='cuda',
                         help='device(s) used to train or test or infer')
    parser_.add_argument("--num_classes", type=int, default=3,
                         help="This refers to the number of classes in the segmentation mode\
                         , so either 2(Merge) or 3(GGO+CO)")
    parser_.add_argument("--model_name", type=str, default='U2Net_n',help='model name')
    parser_.add_argument('--normalize', type=bool,
                         default=False, help='whether used z-score to preprocess the imgs')
    if stage == "train":
        parser_.add_argument("--batch_size", type=int,
                             default=1, help='batch_size')
        parser_.add_argument("--start_epoch", type=str, default=1,help='begin epoch(0 or 1), when u retrain the model, ignore it')
        parser_.add_argument("--num_epochs", type=int, default=100,help='how epoch u want to train')
        parser_.add_argument("--save_dir", type=str, default="./output/saved_models",
                             help="Directory to save pth or onnx")

        parser_.add_argument("--train_data_dir", type=str, default='./data/seg/train',
                             help="Path to the training data. Must contain imgs and masks folder")
        parser_.add_argument("--val_data_dir", type=str, default='./data/seg/test',
                             help="Path to the validation data. Must contain imgs and masks folder")
        parser_.add_argument("--save_every", type=int, default=10,help='epoch%save_every==0->save')
        parser_.add_argument("--lrate", type=float,
                             default=1e-3, help="init Learning rate")
        parser_.add_argument('--log_name', type=str,
                             default=None, help='中断后继续训练记载')
        parser_.add_argument('--weight', type=float, nargs='+',
                             default=[1, 20, 20], help='交叉熵权值')

    elif stage == "test":

        parser_.add_argument("--test_data_dir", type=str, default='./data/seg/test/',
                             help="Path to the test data. Must contain imgs and masks folder")
        parser_.add_argument("--save_seg", type=str,
                             default='./output/segResult/',
                             help='path to save the output masks')
    elif stage == "infer":

        parser_.add_argument("--infer_data_dirs", type=str, nargs='+', default=['/home/e201cv/Desktop/covid_data/process_clf/train',
                                                                                '/home/e201cv/Desktop/covid_data/process_clf/val',
                                                                                 '/home/e201cv/Desktop/covid_data/process_clf/test'],
                             help="Path to the infer data. Must contain imgs folder")
    model_args = parser_.parse_args()
    return model_args
