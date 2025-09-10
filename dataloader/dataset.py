import os
import cv2
from torch.utils.data import Dataset


class MedicalDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",#训练集
        transform=None,#数据增强
        train_file_dir="train.txt",#包含训练数据样本名称的文件
        val_file_dir="val.txt",#指向包含验证数据样本名称的文件
    ):
        

        self._base_dir = base_dir
        self.sample_list = []# 初始化一个空列表，用于存储样本文件的名称。
        self.split = split#将分割类型（train或val）赋值给实例属性
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                #这个文件的路径是由 self._base_dir（基本目录）和 train_file_dir 组合而成。在文件中，应该存放着所有训练样本的名称，每一行代表一个样本。
                #这一行的作用是打开上述拼接出来的文件路径，并以只读模式（"r"）打开文件。
                self.sample_list = f1.readlines()
                #使用 readlines() 方法将文件中所有的行读入到 self.sample_list 列表中。此时，每一个元素都是一个字符串，代表一行内容（即一个样本名）
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]#去掉字符串末尾的换行符 \n。处理后，列表中的每个元素就变成了干净的样本文件名
#将字符串中的换行符\n替换为空字符串

        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)#获取这个列表中的元素数量，

    def __getitem__(self, idx):

        case = self.sample_list[idx]
        #通过索引idx从self.sample_list中获取当前样本的名称，case将包含该样本的文件名（不含路径）

        image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
        # print("Image path:", image )
        #case + '.png' 是图像文件的名称，case 变量从 self.sample_list 中提取，代表当前样本的文件名，不包含扩展名。+ '.png' 则为该文件添加了 .png 扩展名
        #其中图像的名字由case变量提供，并添加了.png扩展名。
        # label = cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '_mask.png'), cv2.IMREAD_GRAYSCALE)[..., None]
        label = cv2.imread(os.path.join(self._base_dir, 'masks', '0', case+ '.png' ), cv2.IMREAD_GRAYSCALE)[..., None]
        # print("Label path:", label )
        #第一个参数为文件路径，第二个参数 cv2.IMREAD_GRAYSCALE 指示读取图像时将其转换为灰度图像。
        #在读取的灰度图像的最后增加一个维度
        #将灰度图像（二维数组）转换为一个三维数组，形状为 (高度, 宽度, 1)，

        augmented = self.transform(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32') / 255#将像素值归一化到0到1之间
        image = image.transpose(2, 0, 1)#调整为（通道，高度，宽度
        #即从(高度, 宽度, 通道)调整为(通道, 高度, 宽度)。

        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        sample = {"image": image, "label": label, "case": case}
        #归一化并调整维度后的图像数据
        #归一化并调整维度后的标签数据
        #样本的文件名，提供了实际图像对应的标识信息
        return sample
    #将这个字典返回，这样可以在后续的训练或验证过程中使用。
