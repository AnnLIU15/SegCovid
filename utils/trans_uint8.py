import argparse
from glob import glob
from multiprocessing import Process

import numpy as np


def imgs_uint8(data_path):
    files = glob(data_path+'/*.npy')
    print('begin process ', data_path)
    for file in files:
        pic_=np.load(file)
        
        pic_=pic_.astype(np.uint8)
        np.save(file,pic_)
    print('end process ', data_path)


def main(args):
    in_dir = args.in_dir
    if isinstance(in_dir, str):
        in_dir = [in_dir]
    process_list = []
    for idx, _ in enumerate(in_dir):
        process_list.append(Process(target=imgs_uint8,args=(in_dir[idx],)))
    for process in process_list:
        process.start()


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--in_dir', type=str, nargs='+',
                         default='/home/e201cv/Desktop/covid_seg/data/clf/train/masks')
    args = parser_.parse_args()
    main(args)
