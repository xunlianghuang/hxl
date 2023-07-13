import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
class datasets:
    def __init__(self, path, mode='Training',transform=None):
        self.path=path
        self.mode=mode
        self.transform=None
        self.image_array=None
        self.image_label=transform
        self.transform=transform
        self.init_img()

    def init_img(self):
        if self.transform==None:
            #print('这一行有没有运行')
            self.transform=transforms.ToTensor()

        fer2013 = pd.read_csv(self.path)
        fer2013 = fer2013[fer2013['Usage']==self.mode]#fer2013['Usage']==self.mode找索引

        self.image_array = np.zeros(shape=(len(fer2013), 48, 48))
        self.image_label = np.array(list(map(int, fer2013['emotion'].values)))

        for i, row in enumerate(fer2013.index):
            image = np.fromstring(fer2013.loc[row, 'pixels'], dtype=int, sep=' ')
            image = np.reshape(image, (48, 48))
            self.image_array[i] = image

    def __len__(self):
        return len(self.image_array)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            print('不知道为什么会运行这行代码')
            idx = item.tolist()
        img = np.array(self.image_array[item])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = torch.tensor(self.image_label[item]).type(torch.long)
        sample = (img, label)
        return sample


def get_dataloaders(path='datasets/fer2013/fer2013.csv',bs=64):
    mu, st = 0, 255
    train_transform = transforms.Compose([
        # 随机裁剪
        #transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
        # 左右翻转
        transforms.RandomHorizontalFlip(),
        # 随机旋转
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),


        transforms.ToTensor(),
        # 归一化
        transforms.Normalize(mean=(mu,), std=(st,)),

    ])
    test_transform = transforms.Compose([
        # transforms.Scale(52),
        transforms.ToTensor(),
        transforms.Normalize(mean=(mu,), std=(st,)),
    ])
    train=datasets(path=path,mode="Training",transform=train_transform)
    val =datasets(path=path,mode="PrivateTest",transform=test_transform)
    test=datasets(path=path,mode="PublicTest",transform=test_transform)
    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=0)
    valloader = DataLoader(val, batch_size=64, shuffle=True, num_workers=0)
    testloader = DataLoader(test, batch_size=64, shuffle=True, num_workers=0)
    return trainloader, valloader, testloader
if __name__ == "__main__":

    train_loader,_1,_2 = get_dataloaders('..\datasets/fer2013/fer2013.csv')
    for image, label in train_loader:
        print(image.shape)
        print(label)
    for image, label in _1:
        print(image.shape)
        print(label)
    for image, label in _2:
        print(image.shape)
        print(label)
