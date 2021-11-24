from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
from data_dict import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from models.vanilla_vae import VanillaVAE as VAE

class GeneDataset(Dataset):
    #传递数据路径，csv路径 ，数据增强方法
    def __init__(self, dir_path='',csv='', transform=None, target_transform=None,latent_dim=5,index=0):
        super(GeneDataset, self).__init__()
        self.index=index
        #一个个往列表里面加绝对路径
        '''self.path = []
        #读取csv
        self.data = pd.read_csv(csv)
        #对标签进行硬编码，例如0 1 2 3 4，把字母变成这个
        colorMap = {elem: index + 1 for index, elem in enumerate(set(self.data["label"]))}
        self.data['label'] = self.data['label'].map(colorMap)
        #创造空的label准备存放标签
        self.num = int(self.data.shape[0])  # 一共多少照片
        self.label = np.zeros(self.num, dtype=np.int32)
        #迭代得到数据路径和标签一一对应
        for index, row in self.data.iterrows():
            self.path.append(os.path.join(dir_path,row['filename']))
            self.label[index] = row['label']  # 将数据全部读取出来
        #训练数据增强
        self.transform = transform
        #验证数据增强在这里没用
        self.target_transform = target_transform'''
        #####################
        train_file = "data_all.txt"
        label_file = "label_all.txt"
        self.train_data = pd.read_table(train_file, index_col=0)
        self.train_label = pd.read_table(label_file, index_col=0).values.ravel()
        data_dict = {'origin_data': origin_data, 'square_data': square_data, 'log_data': log_data,
                     'radical_data': radical_data, 'cube_data': cube_data}

        platform = "platform.json"
        data_type = "origin_data"
        model_type = "VAE"
        with open(platform, 'r') as f:
            gene_dict = json.load(f)
            f.close()

        count = 0
        num = len(gene_dict)
        gene_list = []
        print('Now start training gene...')

        data_train = data_dict[data_type](self.train_data)
        for i,gene in enumerate(gene_dict):
            count += 1
            self.gene_data_train = []
            if self.index == i:
                for residue in data_train.index:
                    if residue in gene_dict[gene]:
                        self.gene_data_train.append(data_train.loc[residue])
                if len(self.gene_data_train) == 0:
                    print('Contained Nan data, has been removed!')
                    continue

                self.gene_data_train = np.array(self.gene_data_train)
                gene_list.append(gene)
                print('Now training ' + gene + "\tusing " + model_type + "\ton " + data_type + "\t" + str(
                    int(count * 100 / num)) + '% ...')
                model_dict = {'LinearRegression': LinearRegression, 'LogisticRegression': LogisticRegression,
                              'L1': Lasso, 'L2': Ridge, 'VAE': VAE(len(gene), latent_dim)}


    #最关键的部分，在这里使用前面的方法
    def __getitem__(self, index):
        '''img =Image.open(self.path[index]).convert('RGB')
        labels = self.label[index]
        #在这里做数据增强
        if self.transform is not None:
            img = self.transform(img)  # 转化tensor类型
        return img, labels'''
        return self.gene_data_train,self.train_label

    def __len__(self):
        return len(self.train_label)
