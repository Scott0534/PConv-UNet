import os
import random
import argparse

from glob import glob
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="busi", help='dataset_name')
#数据集的名字
parser.add_argument('--dataset_root', type=str, default="./data", help='dir')
#用于指定数据集的根目录
args = parser.parse_args()
#解析传入的命令行参数，并将结果存储在args对象中

if __name__ == '__main__':

    name = args.dataset_name
    root = os.path.join(args.dataset_root, args.dataset_name)#完整路径

    img_ids = glob(os.path.join(root, 'images', '*.png'))
    #则最终生成的路径将是"./data/busi/images/*.png"。
    #它会查找上述生成的路径中所有以PNG格式结尾的文件。返回的结果是一个文件路径的列表。
    #从指定的数据集目录中提取所有PNG图像文件的路径，并将这些路径存储在img_ids列表中
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    #列表推导式
    #即去除路径，只保留文件名及其后缀
    #如果p是"./data/busi/images/image1.png"，那么os.path.basename(p)将返回"image1.png"
    #函数将文件名拆分成两部分：文件名本身（不包括扩展名）和扩展名的元组。
    #os.path.splitext('image1.png')将返回('image1', '.png')
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=46)
    #random.randint(0, 1024)生成一个在0到1024之间的随机整数作为种子，保证每次分割的可重复性

    with open(os.path.join(root, '{}_train46.txt'.format(name)), 'w') as file:
        for i in train_img_ids:
            file.write(i + '\n')
    print("build train file successfully, path is: {}".format(os.path.join(root, '{}_train46.txt'.format(name))))

    with open(os.path.join(root, '{}_val46.txt'.format(name)), 'w') as file:
        for i in val_img_ids:
            file.writelines(i + '\n')
    print("build validate file successfully, path is: {}".format(os.path.join(root, '{}_val46.txt'.format(name))))

