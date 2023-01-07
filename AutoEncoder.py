import os
import json
import torch
import math
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
from torch import nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torch.nn.utils.prune as prune

import pickle
from time import time

#from AE import *

#tensorboard visualization
def origin_data(data):
    return data


def square_data(data):
    return data ** 2


def log_data(data):
    return np.log(data + 1e-5)


def radical_data(data):
    return data ** (1 / 2)


def cube_data(data):
    return data ** 3
'''
num_epochs = 50
batch_size = 100
hidden_size = 30


# MNIST dataset
dataset = dsets.MNIST(root='../data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
'''
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Autoencoder(nn.Module):
    def __init__(self, in_dim=784, h_dim=400,platform = "platform.json",X_train=pd.read_table("data_train.txt", index_col=0),data_type="origin_data",model_type="AE"):
        super(Autoencoder, self).__init__()
        mid_dim=int(math.sqrt(h_dim * in_dim))
        q1_dim=int(math.sqrt(h_dim * mid_dim))
        q3_dim=int(math.sqrt(mid_dim * in_dim))
        # nn.Linear(q3_dim, mid_dim),
        # nn.ReLU(),
        # nn.Linear(mid_dim, q1_dim),
        # nn.ReLU(),
        if False:
            with open(platform, 'r') as f:
                gene_dict = json.load(f)
                f.close()

            data_dict = {'origin_data': origin_data, 'square_data': square_data, 'log_data': log_data,
                         'radical_data': radical_data, 'cube_data': cube_data}
            data_train = data_dict[data_type](X_train)

            gene_data_train = []
            residuals_name = []
            model = None
            count = 0
            num = len(gene_dict)
            gene_list = []
            for (i, gene) in enumerate(gene_dict):
                count += 1
                # gene_data_train = []
                # residuals_name = []
                for residue in data_train.index:
                    if residue in gene_dict[gene]:
                        gene_data_train.append(data_train.loc[residue])
                        residuals_name.append(residue)
                if len(gene_data_train) == 0:
                    # print('Contained Nan data, has been removed!')
                    continue
                # gene_data_train = np.array(gene_data_train)
                gene_list.append(gene)
                #print('No.' + str(i) + 'inside auto-encoder ' + gene + "\tusing " + model_type + "\ton " + data_type + "\t" + str(
                    #int(count * 100 / num)) + '% ...')
                #print('finish!')

            #print("count=%d" % count )
            #print("gene_list is ")
            #print(gene_list)
            print("len(gene_list)")
            print(len(gene_list))


        self.encoder = nn.Sequential(
            nn.Linear(in_dim, q3_dim),

            nn.ReLU(),
            nn.Linear(q3_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, q1_dim),
            nn.ReLU(),  # nn.Sigmoid()
            nn.Linear(q1_dim, h_dim),
            nn.ReLU()#nn.Sigmoid()
        )

        # nn.Linear(q1_dim, mid_dim),
        # nn.ReLU(),
        # nn.Linear(mid_dim, q3_dim),
        # nn.ReLU(),


        self.decoder = nn.Sequential(

            #prune.custom_from_mask(
            #    nn.Linear(h_dim, mid_dim),name='activation', mask=torch.tensor(np.ones((mid_dim, h_dim))) #'embedding_to_pathway' #np.random.randint(0,2,(q1_dim, h_dim))
            #),

            nn.Linear(h_dim, mid_dim),
            nn.ReLU(),#nn.Sigmoid()
            nn.Linear(mid_dim, in_dim),
            #prune.custom_from_mask(
            #    nn.Linear(mid_dim, in_dim), name='weight',
            #    mask=torch.tensor(np.ones((in_dim, mid_dim)))#'pathway_to_gene'
            #),

            #nn.ReLU(),
            #nn.Linear(mid_dim, q3_dim),
            #nn.ReLU(),
            #nn.Linear(q3_dim, in_dim),
            nn.Sigmoid()
            )


    def forward(self, x):
        """
        Note: image dimension conversion will be handled by external methods
        """
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    def code(self,x):
        out = self.encoder(x)
        return out


class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out output to deeper layer，out_2 as input to next layer，
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2


class MeiNN_DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MeiNN_DownsampleLayer, self).__init__()
        latent_dim=out_ch
        in_dim=in_ch
        mid_dim = int(math.sqrt(latent_dim * in_dim))
        q1_dim = int(math.sqrt(latent_dim * mid_dim))
        q3_dim = int(math.sqrt(mid_dim * in_dim))
        encoder_dims = [in_dim,q3_dim,mid_dim,q1_dim, latent_dim]
        self.encoder = nn.Sequential(
            nn.Linear(encoder_dims[0], encoder_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_dims[1], encoder_dims[2]),
            nn.ReLU(),
            nn.Linear(encoder_dims[2], encoder_dims[3]),
            nn.ReLU(),  # nn.Sigmoid()
            nn.Linear(encoder_dims[3], encoder_dims[4]),
            nn.ReLU()   #nn.Sigmoid()
        )
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out output to deeper layer，out_2 as input to next layer，
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2



class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()

        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: input convolution layer
        :param out: cat with upSampling layer
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out



class MeiNN_UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(MeiNN_UpSampleLayer, self).__init__()

        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: input convolution layer
        :param out: cat with upSampling layer
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        #downSampling
        self.d1=DownsampleLayer(3,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #upSampling
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #输出
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],3,3,1,1),
            nn.Sigmoid(),
            # BCELoss
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out

class MeiNN_UNet(nn.Module):
    def __init__(self,in_dim,gene_layer_dim,latent_dim):
        super(MeiNN_UNet, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]

        mid_dim = int(math.sqrt(latent_dim * in_dim))
        q1_dim = int(math.sqrt(latent_dim * mid_dim))
        q3_dim = int(math.sqrt(mid_dim * in_dim))
        encoder_dims = [in_dim,q3_dim,mid_dim,q1_dim, latent_dim]
        decoder_dims = [latent_dim,gene_layer_dim,in_dim]
        self.encoder = nn.Sequential(
            nn.Linear(encoder_dims[0], encoder_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_dims[1], encoder_dims[2]),
            nn.ReLU(),
            nn.Linear(encoder_dims[2], encoder_dims[3]),
            nn.ReLU(),  # nn.Sigmoid()
            nn.Linear(encoder_dims[3], encoder_dims[4]),
            nn.ReLU()   #nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(decoder_dims[0], decoder_dims[1]),
            nn.ReLU(),
            nn.Linear(decoder_dims[1], decoder_dims[2]),
            nn.Sigmoid()#nn.Tanh()
            )
        #downSampling
        self.d1=DownsampleLayer(3,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #upSampling
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #输出
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],3,3,1,1),
            nn.Sigmoid(),
            # BCELoss
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out

class MeiNN(nn.Module):
    def __init__(self, config, path, date, code, X_train, y_train, platform, model_type, data_type,
                 HIDDEN_DIMENSION, toTrainMeiNN, AE_epoch_from_main=1000, NN_epoch_from_main=1000,
                 separatelyTrainAE_NN=True,
                 model_dir='./saved_model/',
                 train_dataset_filename=r"./dataset/data_train.txt", train_label_filename=r"./dataset/label_train.txt",
                 gene_to_site_dir=r"./platform.json", gene_to_residue_or_pathway_info=None,
                 toAddGeneSite=True, toAddGenePathway=True,
                 multiDatasetMode="softmax", datasetNameList=[], lossMode='reg_mean',skip_connection_mode="unet"):
        super(MeiNN, self).__init__()
        self.outputSet = []
        # self.modelSet=[]
        self.model_dir = model_dir
        self.config = config
        self.built = False
        self.compiled = False
        self.isfit = False
        self.l_rate = 0.01 #K.variable(0.01)
        self.genes = None
        self.classes = None
        self.dot_weights = 0
        # self.gpu_count = K.tensorflow_backend._get_available_gpus()

        # self.gpu_count = tf.config.list_physical_devices('GPU')
        self.x_train = X_train  # pd.read_table(train_dataset_filename,index_col=0)
        self.y_train = y_train  # pd.read_table(train_label_filename, index_col=0).values.ravel()
        self.NN_epoch_from_main = NN_epoch_from_main
        self.AE_epoch_from_main = AE_epoch_from_main
        self.path = path
        self.date = date
        self.code = code
        self.model_type = model_type
        self.data_type = data_type
        self.HIDDEN_DIMENSION = HIDDEN_DIMENSION
        self.gene_to_site_dir = gene_to_site_dir
        self.gene_to_residue_or_pathway_info = gene_to_residue_or_pathway_info
        self.toAddGeneSite = toAddGeneSite
        self.toAddGenePathway = toAddGenePathway
        self.multiDatasetMode = multiDatasetMode
        self.datasetNameList = datasetNameList
        self.lossMode = lossMode
        self.separatelyTrainAE_NN = separatelyTrainAE_NN
        gene_layer_dim = len(self.gene_to_residue_or_pathway_info.gene_to_id_map)
        residue_layer_dim = len(self.gene_to_residue_or_pathway_info.residue_to_id_map)

        in_dim = residue_layer_dim  # int(809)#modified 2022-4-14
        latent_dim = self.gene_to_residue_or_pathway_info.gene_pathway.shape[0]  # self.HIDDEN_DIMENSION
        encoder_shape = [gene_layer_dim, residue_layer_dim, latent_dim]
        decoder_shape = [latent_dim, gene_layer_dim]  # modified on 2022-4-14 #, residue_layer_dim]
        input_shape = (residue_layer_dim)

        mid_dim=int(math.sqrt(latent_dim * in_dim))
        q1_dim=int(math.sqrt(latent_dim * mid_dim))
        q3_dim=int(math.sqrt(mid_dim * in_dim))
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, q3_dim),
            nn.ReLU(),
            nn.Linear(q3_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, q1_dim),
            nn.ReLU(),  # nn.Sigmoid()
            nn.Linear(q1_dim, latent_dim),
            nn.ReLU()#nn.Sigmoid()
        )

        # nn.Linear(q1_dim, mid_dim),
        # nn.ReLU(),
        # nn.Linear(mid_dim, q3_dim),
        # nn.ReLU(),
        self.decoder = nn.Sequential(
            #prune.custom_from_mask(
            #    nn.Linear(h_dim, mid_dim),name='activation', mask=torch.tensor(np.ones((mid_dim, h_dim))) #'embedding_to_pathway' #np.random.randint(0,2,(q1_dim, h_dim))
            #),
            nn.Linear(latent_dim, gene_layer_dim),
            nn.ReLU(),#nn.Sigmoid()
            nn.Linear(gene_layer_dim, residue_layer_dim),
            #prune.custom_from_mask(
            #    nn.Linear(mid_dim, in_dim), name='weight',
            #    mask=torch.tensor(np.ones((in_dim, mid_dim)))#'pathway_to_gene'
            #),

            #nn.ReLU(),
            #nn.Linear(mid_dim, q3_dim),
            #nn.ReLU(),
            #nn.Linear(q3_dim, in_dim),
            nn.Sigmoid()#nn.Tanh()
            )

        in_dim = latent_dim
        # output dimension is 1
        out_dim=1
        if self.multiDatasetMode=="softmax":
            out_dim = len(self.datasetNameList)
        elif self.multiDatasetMode=="multi-task" or self.multiDatasetMode=="pretrain-finetune":
            out_dim = 1

        mid_dim = int(math.sqrt(in_dim * out_dim))
        q3_dim = int(math.sqrt(in_dim * mid_dim))

        q1_dim = int(math.sqrt(latent_dim * mid_dim))
        #self.FCN=[]

        #for i in range(len(self.datasetNameList)):
        self.FCN1=nn.Sequential(
                nn.Linear(in_dim, q3_dim),
                nn.ReLU(),
                nn.Linear(q3_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, q1_dim),
                nn.ReLU(),  # nn.Sigmoid()
                # nn.Linear(q1_dim, q3_dim),
                # nn.ReLU(),
                nn.Linear(q1_dim, out_dim),
                nn.Sigmoid()
            )
        self.FCN2 = nn.Sequential(nn.Linear(in_dim, q3_dim),nn.ReLU(),
                                  nn.Linear(q3_dim, mid_dim),nn.ReLU(),
                                  nn.Linear(mid_dim, q1_dim),nn.ReLU(),  # nn.Sigmoid()
                                  nn.Linear(q1_dim, out_dim),nn.Sigmoid())
        self.FCN3 = nn.Sequential(nn.Linear(in_dim, q3_dim), nn.ReLU(),
                                  nn.Linear(q3_dim, mid_dim), nn.ReLU(),
                                  nn.Linear(mid_dim, q1_dim), nn.ReLU(),  # nn.Sigmoid()
                                  nn.Linear(q1_dim, out_dim), nn.Sigmoid())
        self.FCN4 = nn.Sequential(nn.Linear(in_dim, q3_dim), nn.ReLU(),
                                  nn.Linear
                                  (q3_dim, mid_dim), nn.ReLU(),
                                  nn.Linear(mid_dim, q1_dim), nn.ReLU(),  # nn.Sigmoid()
                                  nn.Linear(q1_dim, out_dim), nn.Sigmoid())
        self.FCN5 = nn.Sequential(nn.Linear(in_dim, q3_dim), nn.ReLU(),
                                  nn.Linear(q3_dim, mid_dim), nn.ReLU(),
                                  nn.Linear(mid_dim, q1_dim), nn.ReLU(),  # nn.Sigmoid()
                                  nn.Linear(q1_dim, out_dim), nn.Sigmoid())
        self.FCN6 = nn.Sequential(nn.Linear(in_dim, q3_dim), nn.ReLU(),
                                  nn.Linear(q3_dim, mid_dim), nn.ReLU(),
                                  nn.Linear(mid_dim, q1_dim), nn.ReLU(),  # nn.Sigmoid()
                                  nn.Linear(q1_dim, out_dim), nn.Sigmoid())


    def forward(self, x):
        """
        Note: image dimension conversion will be handled by external methods
        """
        embedding = self.encoder(x)
        out = self.decoder(embedding)
        #FCN0=self.FCN[0]
        pred = self.FCN1(embedding)
        pred_list=[]
        if self.multiDatasetMode=="multi-task" or self.multiDatasetMode=="pretrain-finetune":
            if len(self.datasetNameList)==6:
                '''for i in range(len(self.datasetNameList)):
                    FCN_i=self.FCN[i]
                    pred_list.append(FCN_i(embedding))'''
                pred1= self.FCN1(embedding)
                pred2 = self.FCN2(embedding)
                pred3 = self.FCN3(embedding)
                pred4 = self.FCN4(embedding)
                pred5 = self.FCN5(embedding)
                pred6 = self.FCN6(embedding)
                pred_list=[pred1,pred2,pred3,pred4,pred5,pred6]
                #pred_list=torch.cat([pred1,pred2,pred3,pred4,pred5,pred6],dim=1)
                return out,pred_list,embedding
                #return out,[pred1,pred2,pred3,pred4,pred5,pred6],embedding
        return out,pred,embedding
    def code(self,x):
        out = self.encoder(x)
        return out


class NN(nn.Module):
    def __init__(self, in_dim=784, h_dim=400):
        super(NN, self).__init__()
        mid_dim=int(math.sqrt(h_dim * in_dim))
        q1_dim=int(math.sqrt(h_dim * mid_dim))
        q3_dim=int(math.sqrt(mid_dim * in_dim))
        # nn.Linear(q3_dim, mid_dim),
        # nn.ReLU(),
        # nn.Linear(mid_dim, q1_dim),
        # nn.ReLU(),
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, q3_dim),
            nn.ReLU(),
            nn.Linear(q3_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, q1_dim),
            nn.ReLU(),  # nn.Sigmoid()
            #nn.Linear(q1_dim, q3_dim),
            #nn.ReLU(),
            nn.Linear(q1_dim, 1),
            nn.Sigmoid()
            )




    def forward(self, x):
        """
        Note: image dimension conversion will be handled by external methods
        """
        out = self.encoder(x)
        #out = self.decoder(out)
        return out
    def code(self,x):
        out = self.encoder(x)
        return out

'''
ae = Autoencoder(in_dim=
                 784, h_dim=hidden_size)

if torch.cuda.is_available():
    ae.cuda()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
iter_per_epoch = len(data_loader)
data_iter = iter(data_loader)

# save fixed inputs for debugging
fixed_x, _ = next(data_iter)
mydir = 'E:/JI/4 SENIOR/2021 fall/VE490/ReGear-gyl/ReGear/data/'
myfile = 'real_image.png'
images_path = os.path.join(mydir, myfile)
torchvision.utils.save_image(Variable(fixed_x).data.cpu(), images_path)
fixed_x = to_var(fixed_x.view(fixed_x.size(0), -1))

for epoch in range(num_epochs):
    t0 = time()
    for i, (images, _) in enumerate(data_loader):

        # flatten the image
        images = to_var(images.view(images.size(0), -1))
        out = ae(images)
        loss = criterion(out, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Time: %.2fs'
                %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.item(), time()-t0))#最初loss.item()位置是loss.data[0]

    # save the reconstructed images
    reconst_images = ae(fixed_x)
    reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
    mydir = 'E:/JI/4 SENIOR/2021 fall/VE490/ReGear-gyl/ReGear/data/'
    myfile = 'reconst_images_%d.png' % (epoch+1)
    reconst_images_path = os.path.join(mydir, myfile)
    torchvision.utils.save_image(reconst_images.data.cpu(),reconst_images_path )


'''
