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


# from AE import *

# tensorboard visualization
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#from torch.nn import functional as F
class MaskedLinear(nn.Linear):
    def __init__(self, *args, mask, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask
        

    def forward(self, input,toDebug=False,toPrintAllWeight=False):
        if toDebug:
            print("^"*100)
            #print(self.weight) 
            print("self.weight.shape")
            print(self.weight.shape)
            print("self.bias.shape")
            print(self.bias.shape)
            print("self.mask.shape")
            print(self.mask.shape)

        self.masked_weight=self.weight.to(device)*self.mask.to(device)

        if toDebug:
            print("self.weight*self.mask.shape")
            print(self.masked_weight.shape)
            if toPrintAllWeight:
                print(self.masked_weight)
                print(torch.nn.functional.linear(input, self.masked_weight, bias=self.bias))
            print(torch.nn.functional.linear(input, self.masked_weight, bias=self.bias).shape)
            #fun=nn.Linear(self.in_features,self.out_features).to(device)
            #print("fun(input).shape")
            #print(fun(input).shape)
            print("^"*100)
        
        return torch.nn.functional.linear(input, self.masked_weight, bias=self.bias)
        #return torch.nn.functional.linear(input, self.weight, bias=self.bias)*self.mask
        #return fun(input).to(device)*self.mask.to(device)
        
class Autoencoder(nn.Module):
    def __init__(self, in_dim=784, h_dim=400, platform="platform.json",
                 X_train=pd.read_table("data_train.txt", index_col=0), data_type="origin_data", model_type="AE"):
        super(Autoencoder, self).__init__()
        mid_dim = int(math.sqrt(h_dim * in_dim))
        q1_dim = int(math.sqrt(h_dim * mid_dim))
        q3_dim = int(math.sqrt(mid_dim * in_dim))
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
                # print('No.' + str(i) + 'inside auto-encoder ' + gene + "\tusing " + model_type + "\ton " + data_type + "\t" + str(
                # int(count * 100 / num)) + '% ...')
                # print('finish!')

            # print("count=%d" % count )
            # print("gene_list is ")
            # print(gene_list)
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
            nn.ReLU()  # nn.Sigmoid()
        )

        # nn.Linear(q1_dim, mid_dim),
        # nn.ReLU(),
        # nn.Linear(mid_dim, q3_dim),
        # nn.ReLU(),

        self.decoder = nn.Sequential(

            # prune.custom_from_mask(
            #    nn.Linear(h_dim, mid_dim),name='activation', mask=torch.tensor(np.ones((mid_dim, h_dim))) #'embedding_to_pathway' #np.random.randint(0,2,(q1_dim, h_dim))
            # ),

            nn.Linear(h_dim, mid_dim),
            nn.ReLU(),  # nn.Sigmoid()
            nn.Linear(mid_dim, in_dim),
            # prune.custom_from_mask(
            #    nn.Linear(mid_dim, in_dim), name='weight',
            #    mask=torch.tensor(np.ones((in_dim, mid_dim)))#'pathway_to_gene'
            # ),

            # nn.ReLU(),
            # nn.Linear(mid_dim, q3_dim),
            # nn.ReLU(),
            # nn.Linear(q3_dim, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Note: image dimension conversion will be handled by external methods
        """
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def code(self, x):
        out = self.encoder(x)
        return out


class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param x:
        :return: out output to deeper layer，out_2 as input to next layer，
        """
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2


class MeiNN_DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MeiNN_DownsampleLayer, self).__init__()
        latent_dim = out_ch
        in_dim = in_ch
        mid_dim = int(math.sqrt(latent_dim * in_dim))
        q1_dim = int(math.sqrt(latent_dim * mid_dim))
        q3_dim = int(math.sqrt(mid_dim * in_dim))
        encoder_dims = [in_dim, q3_dim, mid_dim, q1_dim, latent_dim]
        self.encoder = nn.Sequential(
            nn.Linear(encoder_dims[0], encoder_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_dims[1], encoder_dims[2]),
            nn.ReLU(),
            nn.Linear(encoder_dims[2], encoder_dims[3]),
            nn.ReLU(),  # nn.Sigmoid()
            nn.Linear(encoder_dims[3], encoder_dims[4]),
            nn.ReLU()  # nn.Sigmoid()
        )
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param x:
        :return: out output to deeper layer，out_2 as input to next layer，
        """
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2


class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()

        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out):
        '''
        :param x: input convolution layer
        :param out: cat with upSampling layer
        :return:
        '''
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out


class MeiNN_UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(MeiNN_UpSampleLayer, self).__init__()

        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out):
        '''
        :param x: input convolution layer
        :param out: cat with upSampling layer
        :return:
        '''
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels = [2 ** (i + 6) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # downSampling
        self.d1 = DownsampleLayer(3, out_channels[0])  # 3-64
        self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
        self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
        self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
        # upSampling
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
        self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
        # 输出
        self.o = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], 3, 3, 1, 1),
            nn.Sigmoid(),
            # BCELoss
        )

    def forward(self, x):
        out_1, out1 = self.d1(x)
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)
        out5 = self.u1(out4, out_4)
        out6 = self.u2(out5, out_3)
        out7 = self.u3(out6, out_2)
        out8 = self.u4(out7, out_1)
        out = self.o(out8)
        return out


class MeiNN_UNet(nn.Module):
    def __init__(self, in_dim, gene_layer_dim, latent_dim):
        super(MeiNN_UNet, self).__init__()
        out_channels = [2 ** (i + 6) for i in range(5)]  # [64, 128, 256, 512, 1024]

        mid_dim = int(math.sqrt(latent_dim * in_dim))
        q1_dim = int(math.sqrt(latent_dim * mid_dim))
        q3_dim = int(math.sqrt(mid_dim * in_dim))
        encoder_dims = [in_dim, q3_dim, mid_dim, q1_dim, latent_dim]
        decoder_dims = [latent_dim, gene_layer_dim, in_dim]
        self.encoder = nn.Sequential(
            nn.Linear(encoder_dims[0], encoder_dims[1]),
            nn.ReLU(),
            nn.Linear(encoder_dims[1], encoder_dims[2]),
            nn.ReLU(),
            nn.Linear(encoder_dims[2], encoder_dims[3]),
            nn.ReLU(),  # nn.Sigmoid()
            nn.Linear(encoder_dims[3], encoder_dims[4]),
            nn.ReLU()  # nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(decoder_dims[0], decoder_dims[1]),
            nn.ReLU(),
            nn.Linear(decoder_dims[1], decoder_dims[2]),
            nn.Sigmoid()  # nn.Tanh()
        )
        # downSampling
        self.d1 = DownsampleLayer(in_dim, out_channels[0])  # 3-64
        self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
        self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
        self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
        # upSampling
        self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
        self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
        self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
        self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
        # 输出
        self.o = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], 3, 3, 1, 1),
            nn.Sigmoid(),
            # BCELoss
        )

    def forward(self, x):
        out_1, out1 = self.d1(x)
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)
        out5 = self.u1(out4, out_4)
        out6 = self.u2(out5, out_3)
        out7 = self.u3(out6, out_2)
        out8 = self.u4(out7, out_1)
        out = self.o(out8)
        return out


class VAE(nn.Module):
    def __init__(self,input_dim,gene_dim,latent_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # mean
        self.fc22 = nn.Linear(400, 20)  # var
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # matrix multiply each other and make these element as exp of e
        eps = torch.FloatTensor(std.size()).normal_()  #generate random array
        if torch.cuda.is_available():
            eps = eps.cuda()
        return eps.mul(std).add_(mu)  # use a normal distribution multiplies stddev, then add mean, make latent vector to normal distribution

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)  # 编码
        z = self.reparametrize(mu, logvar)  # reparamatrize to normal disribution
        return self.decode(z), mu, logvar  # decode, meanwhile output mean and stddev



class MeiNN(nn.Module):
    def __init__(self, config, path, date, code, X_train, y_train, platform, model_type, data_type,
                 HIDDEN_DIMENSION, toTrainMeiNN, AE_epoch_from_main=1000, NN_epoch_from_main=1000,
                 separatelyTrainAE_NN=True,
                 model_dir='./saved_model/',
                 train_dataset_filename=r"./dataset/data_train.txt", train_label_filename=r"./dataset/label_train.txt",
                 gene_to_site_dir=r"./platform.json", gene_to_residue_or_pathway_info=None,
                 toAddGeneSite=True, toAddGenePathway=True,
                 multiDatasetMode="softmax", datasetNameList=[], lossMode='reg_mean', skip_connection_mode="unet&VAE&hdmsk"):
        super(MeiNN, self).__init__()
        self.outputSet = []
        # self.modelSet=[]
        self.model_dir = model_dir
        self.config = config
        self.built = False
        self.compiled = False
        self.isfit = False
        self.l_rate = 0.01  # K.variable(0.01)
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
        self.skip_connection_mode = skip_connection_mode
        gene_layer_dim = len(self.gene_to_residue_or_pathway_info.gene_to_id_map)
        residue_layer_dim = len(self.gene_to_residue_or_pathway_info.residue_to_id_map)

        in_dim = residue_layer_dim  # int(809)#modified 2022-4-14
        latent_dim = self.gene_to_residue_or_pathway_info.gene_pathway.shape[0]  # self.HIDDEN_DIMENSION
        encoder_shape = [gene_layer_dim, residue_layer_dim, latent_dim]
        decoder_shape = [latent_dim, gene_layer_dim]  # modified on 2022-4-14 #, residue_layer_dim]
        input_shape = (residue_layer_dim)

        mid_dim = int(math.sqrt(latent_dim * in_dim))
        q1_dim = int(math.sqrt(latent_dim * mid_dim))
        q3_dim = int(math.sqrt(mid_dim * in_dim))
        mid_dim_u = gene_layer_dim#int(math.sqrt(latent_dim * in_dim))
        q1_dim_u = int(math.sqrt(latent_dim * mid_dim_u))
        q3_dim_u = int(math.sqrt(mid_dim_u * in_dim))
        # if skip_connection_mode=="unet":
        #    self.myMeiNN_UNet=MeiNN_UNet(in_dim,gene_layer_dim,latent_dim)
        print("~"*100)
        print("DEBUG:in MeiNN architecture mode="+skip_connection_mode)
        print("~"*100)
        self.encoder1 = nn.Sequential(
            nn.Linear(in_dim, q3_dim),
            nn.ReLU())
        self.encoder2=nn.Sequential(
            nn.Linear(q3_dim, mid_dim),
            nn.ReLU())
        self.encoder3=nn.Sequential(
            nn.Linear(mid_dim, q1_dim),
            nn.ReLU())  # nn.Sigmoid()
        self.encoder4=nn.Sequential(
            nn.Linear(q1_dim, latent_dim),
            nn.ReLU()  # nn.Sigmoid()
        )
        self.bn_site1 = nn.BatchNorm1d(in_dim)
        self.bn_site2 = nn.BatchNorm1d(in_dim)
        self.bn_site3 = nn.BatchNorm1d(in_dim)
        self.bn_gene1 = nn.BatchNorm1d(gene_layer_dim)
        self.bn_gene2 = nn.BatchNorm1d(gene_layer_dim)
        self.bn_gene3 = nn.BatchNorm1d(gene_layer_dim)
        self.bn_path1 = nn.BatchNorm1d(latent_dim)
        self.bn_path2 = nn.BatchNorm1d(latent_dim)
        self.bn_path3 = nn.BatchNorm1d(latent_dim)
        self.bn_q3_1 = nn.BatchNorm1d(q3_dim)
        self.bn_q3_u = nn.BatchNorm1d(q3_dim_u)
        self.bn_mid1 = nn.BatchNorm1d(mid_dim)
        self.bn_mid_u = nn.BatchNorm1d(mid_dim_u)
        self.bn_q1_1 = nn.BatchNorm1d(q1_dim)
        self.bn_q1_u = nn.BatchNorm1d(q1_dim_u)
        
        self.gene_site_tensor = torch.tensor(self.gene_to_residue_or_pathway_info.gene_to_residue_map, dtype=torch.float).T
        self.pathway_gene_tensor = torch.tensor(self.gene_to_residue_or_pathway_info.gene_pathway.T.values, dtype=torch.float)
        if "hdmsk-4enc-self-fc" in self.skip_connection_mode:
            case_type="hdmsk-4enc-self-fc"
            print("detected"+case_type+" in encoder")
            
            self.encoder1 = nn.Sequential(
                MaskedLinear(residue_layer_dim, gene_layer_dim,mask=self.gene_site_tensor.T),
                nn.ReLU(),  # nn.Sigmoid()
                )
            self.encoder2 = nn.Sequential(
                nn.Linear(gene_layer_dim, gene_layer_dim),
                nn.ReLU(),  # nn.Sigmoid()
                )
            
            self.encoder3 = nn.Sequential(
                MaskedLinear(gene_layer_dim, latent_dim,mask=self.pathway_gene_tensor.T),
                nn.Sigmoid()
                )
            self.encoder4 = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),  # nn.Sigmoid()
                )
            if "VAE" in self.skip_connection_mode:
                    self.encoder4_var = nn.Sequential(
                        nn.Linear(latent_dim, latent_dim),
                        nn.ReLU(),
                    )
            print(case_type+" maskedLinear in encoder defined")
        elif "hdmsk-2enc" in self.skip_connection_mode:
            case_type="hdmsk-2enc"
            print("detected"+case_type+" in encoder")
            
            self.encoder1 = nn.Sequential(
                MaskedLinear(residue_layer_dim, gene_layer_dim,mask=self.gene_site_tensor.T),
                nn.ReLU(),  # nn.Sigmoid()
                )
            #only encoder 1 and 4
            self.encoder4 = nn.Sequential(
                MaskedLinear(gene_layer_dim, latent_dim,mask=self.pathway_gene_tensor.T),
                nn.Sigmoid()
                )
            if "VAE" in self.skip_connection_mode:
                    self.encoder4_var = nn.Sequential(
                        nn.Linear(gene_layer_dim, latent_dim),
                        nn.ReLU(),
                    )
            print(case_type+" maskedLinear in encoder defined")
           
        else:#no hardmask exists in mode
            if "unet" in self.skip_connection_mode:# or "VAE" in self.skip_connection_mode:
                self.encoder1 = nn.Sequential(
                    nn.Linear(in_dim, q3_dim_u),
                    nn.ReLU(),
                )
                self.encoder2 = nn.Sequential(
                    nn.Linear(q3_dim_u, mid_dim_u),
                    nn.ReLU(),
                )

                self.encoder3 = nn.Sequential(
                    nn.Linear(mid_dim_u, latent_dim),
                    nn.ReLU(),
                )

                self.encoder4 = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU(),
                )
                if "VAE" in self.skip_connection_mode:
                    self.encoder4_var = nn.Sequential(
                        nn.Linear(latent_dim, latent_dim),
                        nn.ReLU(),
                    )

            if "unet" not in self.skip_connection_mode and "VAE" in self.skip_connection_mode:
                self.encoder1 = nn.Sequential(
                    nn.Linear(in_dim, q3_dim),
                    nn.ReLU(),
                )
                self.encoder2 = nn.Sequential(
                    nn.Linear(q3_dim, mid_dim),
                    nn.ReLU(),
                )

                self.encoder3 = nn.Sequential(
                    nn.Linear(mid_dim, q1_dim),
                    nn.ReLU(),
                )

                self.encoder4 = nn.Sequential(
                    nn.Linear(q1_dim, latent_dim),
                    nn.ReLU(),
                )
                if "VAE" in self.skip_connection_mode:
                    self.encoder4_var = nn.Sequential(
                        nn.Linear(q1_dim, latent_dim),
                        nn.ReLU(),
                    )
        # nn.Linear(q1_dim, mid_dim),
        # nn.ReLU(),
        # nn.Linear(mid_dim, q3_dim),
        # nn.ReLU(),

        self.kl_divergence = 0
        
        
        
        if "hdmsk" in self.skip_connection_mode:
            print("detected hardmask in decoder")
            #pathway_gene_tensor = torch.tensor(self.gene_to_residue_or_pathway_info.gene_pathway.T.values, dtype=torch.float)
            self.decoder1 = nn.Sequential(
                MaskedLinear(latent_dim, gene_layer_dim,mask=self.pathway_gene_tensor),
                nn.ReLU(),  # nn.Sigmoid()
                )
            #gene_site_tensor = torch.tensor(self.gene_to_residue_or_pathway_info.gene_to_residue_map, dtype=torch.float).T
            self.decoder2 = nn.Sequential(
                MaskedLinear(gene_layer_dim, residue_layer_dim,mask=self.gene_site_tensor),
                nn.Sigmoid()
                )
            print("hard maskedLinear in decoder defined")
        else:
            print("not detected hardmask in decoder")
            self.decoder1 = nn.Sequential(
                    nn.Linear(latent_dim, gene_layer_dim),
                    nn.ReLU(),  # nn.Sigmoid()
                    )
            self.decoder2 = nn.Sequential(
                nn.Linear(gene_layer_dim, residue_layer_dim),
                nn.Sigmoid()
                )
        '''
        self.decoder = nn.Sequential(
                # prune.custom_from_mask(
                #    nn.Linear(h_dim, mid_dim),name='activation', mask=torch.tensor(np.ones((mid_dim, h_dim))) #'embedding_to_pathway' #np.random.randint(0,2,(q1_dim, h_dim))
                # ),
                nn.Linear(latent_dim, gene_layer_dim),
                nn.ReLU(),  # nn.Sigmoid()
                nn.Linear(gene_layer_dim, residue_layer_dim),
                # prune.custom_from_mask(
                #    nn.Linear(mid_dim, in_dim), name='weight',
                #    mask=torch.tensor(np.ones((in_dim, mid_dim)))#'pathway_to_gene'
                # ),

                # nn.ReLU(),
                # nn.Linear(mid_dim, q3_dim),
                # nn.ReLU(),
                # nn.Linear(q3_dim, in_dim),
                nn.Sigmoid()  # nn.Tanh()
                )
        '''
        in_dim_fcn = latent_dim
        # output dimension is 1
        out_dim_fcn = 1
        if self.multiDatasetMode == "softmax":
            out_dim_fcn = len(self.datasetNameList)
        elif self.multiDatasetMode == "multi-task" or self.multiDatasetMode == "pretrain-finetune":
            out_dim_fcn = 1


        mid_dim_fcn = int(math.sqrt(in_dim_fcn * out_dim_fcn))
        q3_dim_fcn = int(math.sqrt(in_dim_fcn * mid_dim_fcn))

        q1_dim_fcn = int(math.sqrt(in_dim_fcn * mid_dim_fcn))#TODO:test difference of in_dim, out_dim
        # self.FCN=[]

        # for i in range(len(self.datasetNameList)):
        self.FCN1 = nn.Sequential(
            nn.Linear(in_dim_fcn, q3_dim_fcn),
            nn.ReLU(),
            nn.Linear(q3_dim_fcn, mid_dim_fcn),
            nn.ReLU(),
            nn.Linear(mid_dim_fcn, q1_dim_fcn),
            nn.ReLU(),  # nn.Sigmoid()
            # nn.Linear(q1_dim, q3_dim),
            # nn.ReLU(),
            nn.Linear(q1_dim_fcn, out_dim_fcn),
            nn.Sigmoid()
        )


        self.FCN2 = nn.Sequential(nn.Linear(in_dim_fcn, q3_dim_fcn), nn.ReLU(),
                                  nn.Linear(q3_dim_fcn, mid_dim_fcn), nn.ReLU(),
                                  nn.Linear(mid_dim_fcn, q1_dim_fcn), nn.ReLU(),  # nn.Sigmoid()
                                  nn.Linear(q1_dim_fcn, out_dim_fcn), nn.Sigmoid())
        self.FCN3 = nn.Sequential(nn.Linear(in_dim_fcn, q3_dim_fcn), nn.ReLU(),
                                  nn.Linear(q3_dim_fcn, mid_dim_fcn), nn.ReLU(),
                                  nn.Linear(mid_dim_fcn, q1_dim_fcn), nn.ReLU(),  # nn.Sigmoid()
                                  nn.Linear(q1_dim_fcn, out_dim_fcn), nn.Sigmoid())
        self.FCN4 = nn.Sequential(nn.Linear(in_dim_fcn, q3_dim_fcn), nn.ReLU(),
                                  nn.Linear
                                  (q3_dim_fcn, mid_dim_fcn), nn.ReLU(),
                                  nn.Linear(mid_dim_fcn, q1_dim_fcn), nn.ReLU(),  # nn.Sigmoid()
                                  nn.Linear(q1_dim_fcn, out_dim_fcn), nn.Sigmoid())
        self.FCN5 = nn.Sequential(nn.Linear(in_dim_fcn, q3_dim_fcn), nn.ReLU(),
                                  nn.Linear(q3_dim_fcn, mid_dim_fcn), nn.ReLU(),
                                  nn.Linear(mid_dim_fcn, q1_dim_fcn), nn.ReLU(),  # nn.Sigmoid()
                                  nn.Linear(q1_dim_fcn, out_dim_fcn), nn.Sigmoid())
        self.FCN6 = nn.Sequential(nn.Linear(in_dim_fcn, q3_dim_fcn), nn.ReLU(),
                                  nn.Linear(q3_dim_fcn, mid_dim_fcn), nn.ReLU(),
                                  nn.Linear(mid_dim_fcn, q1_dim_fcn), nn.ReLU(),  # nn.Sigmoid()
                                  nn.Linear(q1_dim_fcn, out_dim_fcn), nn.Sigmoid())

    def reparametrize(self, mu, logvar):#for VAE
        std = logvar.mul(0.5).exp_()  # matrix multiply each other and make these element as exp of e
        eps = torch.FloatTensor(std.size()).normal_()  #generate random array
        if torch.cuda.is_available():
            eps = eps.cuda()
        return eps.mul(std).add_(mu)  # use a normal distribution multiplies stddev, then add mean, make latent vector to normal distribution

    def kl_divergence_function(self,mu,logvar):
        sigma = logvar.mul(0.5).exp_()
        return (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

    def forward(self, x):
        """
        Note: image dimension conversion will be handled by external methods
        """
        if "hdmsk-4enc-self-fc" in self.skip_connection_mode:
            #normally the encoder and decoder dimension is both defined by site-gene-pathway relation.
            #encoder site-gene-pathway case haven't support unet mode
            x1 = self.encoder1(x)
            if "batchnorm"in self.skip_connection_mode:
                x1 = self.bn_gene1(x1)
            x2 = self.encoder2(x1)
            if "batchnorm"in self.skip_connection_mode:
                x2 = self.bn_gene2(x2)
            x3 = self.encoder3(x2)
            if "batchnorm"in self.skip_connection_mode:
                x3 = self.bn_path1(x3)
            embedding_mu = self.encoder4(x3)
            if "batchnorm"in self.skip_connection_mode:
                embedding_mu = self.bn_path2(embedding_mu)
            
            if "VAE" in self.skip_connection_mode:#VAE:x3 ->(mu,var)
                embedding_logvar = self.encoder4_var(x3)
                #embedding_cat = embedding + x3  # torch.cat((embedding, x3), dim=1)
                embedding = self.reparametrize(embedding_mu, embedding_logvar)  # reparamatrize to normal disribution
            else:#normal AE:x3 ->embedding
                embedding=embedding_mu
            x5 = self.decoder1(embedding)
            if "batchnorm"in self.skip_connection_mode:
                x5 = self.bn_gene3(x5)
            #x5_cat = x5 + x2  # torch.cat((x5, x2), dim=1)
            out = self.decoder2(x5)
            #if "batchnorm"in self.skip_connection_mode:
            #    out = self.bn_site3(out)
        elif "hdmsk-2enc" in self.skip_connection_mode:
            #normally the encoder and decoder dimension is both defined by site-gene-pathway relation.
            #encoder site-gene-pathway case haven't support unet mode
            x1 = self.encoder1(x)
            if "batchnorm"in self.skip_connection_mode:
                x1 = self.bn_gene1(x1)
            embedding_mu = self.encoder4(x1)
            if "batchnorm"in self.skip_connection_mode:
                embedding_mu = self.bn_path1(embedding_mu)
            
            if "VAE" in self.skip_connection_mode:#VAE:x3 ->(mu,var)
                embedding_logvar = self.encoder4_var(x1)
                #embedding_cat = embedding + x3  # torch.cat((embedding, x3), dim=1)
                embedding = self.reparametrize(embedding_mu, embedding_logvar)  # reparamatrize to normal disribution
            else:#normal AE:x3 ->embedding
                embedding=embedding_mu
            x5 = self.decoder1(embedding)
            if "batchnorm"in self.skip_connection_mode:
                x5 = self.bn_gene2(x5)
            #x5_cat = x5 + x2  # torch.cat((x5, x2), dim=1)
            out = self.decoder2(x5)
            #if "batchnorm"in self.skip_connection_mode:
            #    out = self.bn_site2(out)
        elif "unet"in self.skip_connection_mode and "VAE" in self.skip_connection_mode:#unet&VAE#this case ,encoder is not hardmasked
            x1 = self.encoder1(x)
            if "batchnorm"in self.skip_connection_mode:
                x1 = self.bn_q3_u(x1)
            x2 = self.encoder2(x1)
            if "batchnorm"in self.skip_connection_mode:
                x2 = self.bn_mid_u(x2)
            x3 = self.encoder3(x2)
            if "batchnorm"in self.skip_connection_mode:
                x3 = self.bn_path1(x3)
            embedding_mu = self.encoder4(x3)
            if "batchnorm"in self.skip_connection_mode:
                embedding_mu = self.bn_path2(embedding_mu)
            embedding_logvar = self.encoder4_var(x3)
            embedding = self.reparametrize(embedding_mu, embedding_logvar)  # reparamatrize to normal disribution
            self.kl_divergence = self.kl_divergence_function(embedding_mu, embedding_logvar)
            embedding_cat = embedding + x3  # torch.cat((embedding, x3), dim=1)
            x5 = self.decoder1(embedding_cat)
            if "batchnorm"in self.skip_connection_mode:
                x5 = self.bn_gene2(x5)
            x5_cat = x5 + x2  # torch.cat((x5, x2), dim=1)
            out = self.decoder2(x5_cat)
            #if "batchnorm"in self.skip_connection_mode:
            #    out = self.bn_site2(out)
        elif not("unet" in self.skip_connection_mode) and ("VAE" in self.skip_connection_mode): #VAE #this case ,encoder is not hardmasked
            x1 = self.encoder1(x)
            if "batchnorm"in self.skip_connection_mode:
                x1 = self.bn_q3_1(x1)
            x2 = self.encoder2(x1)
            if "batchnorm"in self.skip_connection_mode:
                x2 = self.bn_mid1(x2)
            x3 = self.encoder3(x2)
            if "batchnorm"in self.skip_connection_mode:
                x3 = self.bn_q1_1(x3)
            embedding_mu = self.encoder4(x3)
            if "batchnorm"in self.skip_connection_mode:
                embedding_mu = self.bn_path1(embedding_mu)
            embedding_logvar = self.encoder4_var(x3)
            #embedding_cat = embedding + x3  # torch.cat((embedding, x3), dim=1)
            embedding = self.reparametrize(embedding_mu, embedding_logvar)  # reparamatrize to normal disribution
            x5 = self.decoder1(embedding)
            if "batchnorm"in self.skip_connection_mode:
                x5 = self.bn_gene2(x5)
            #x5_cat = x5 + x2  # torch.cat((x5, x2), dim=1)
            out = self.decoder2(x5)
            #if "batchnorm"in self.skip_connection_mode:
            #    out = self.bn_site2(out)

            #embedding = self.encoder(x)
            #out = self.decoder(embedding)
        elif ("unet" in self.skip_connection_mode) and not("VAE" in self.skip_connection_mode):#unet#this case ,encoder is not hardmasked
            x1 = self.encoder1(x)
            if "batchnorm"in self.skip_connection_mode:
                x1 = self.bn_q3_u(x1)
            x2 = self.encoder2(x1)
            if "batchnorm"in self.skip_connection_mode:
                x2 = self.bn_mid_u(x2)
            x3 = self.encoder3(x2)
            if "batchnorm"in self.skip_connection_mode:
                x3 = self.bn_path1(x3)
            embedding = self.encoder4(x3)
            if "batchnorm"in self.skip_connection_mode:
                embedding = self.bn_path2(embedding)
            embedding_cat=embedding+x3#torch.cat((embedding, x3), dim=1)
            x5 = self.decoder1(embedding_cat)
            if "batchnorm"in self.skip_connection_mode:
                x5 = self.bn_gene2(x5)
            x5_cat=x5+x2#torch.cat((x5, x2), dim=1)
            out = self.decoder2(x5_cat)
            #if "batchnorm"in self.skip_connection_mode:
            #    out = self.bn_site2(out)
        else:# normal auto-encoder, without encoder hardmask,no unet,no VAE
            x1 = self.encoder1(x)
            if "batchnorm"in self.skip_connection_mode:
                x1 = self.bn_q3_1(x1)
            x2 = self.encoder2(x1)
            if "batchnorm"in self.skip_connection_mode:
                x2 = self.bn_mid1(x2)
            x3 = self.encoder3(x2)
            if "batchnorm"in self.skip_connection_mode:
                x3 = self.bn_q1_1(x3)
            embedding = self.encoder4(x3)
            if "batchnorm"in self.skip_connection_mode:
                embedding = self.bn_path1(embedding)

            out_mid = self.decoder1(embedding)#modified to decoder1 and 2 to make format aligned with other moddes
            if "batchnorm"in self.skip_connection_mode:
                out_mid = self.bn_gene2(out_mid)
            out = self.decoder2(out_mid)
            #if "batchnorm"in self.skip_connection_mode:
            #    out = self.bn_site2(out)
        # FCN0=self.FCN[0]
        pred = self.FCN1(embedding)
        pred_list = []
        if self.multiDatasetMode == "multi-task" or self.multiDatasetMode == "pretrain-finetune":
            if len(self.datasetNameList) == 6:
                '''for i in range(len(self.datasetNameList)):
                    FCN_i=self.FCN[i]
                    pred_list.append(FCN_i(embedding))'''
                pred1 = self.FCN1(embedding)
                pred2 = self.FCN2(embedding)
                pred3 = self.FCN3(embedding)
                pred4 = self.FCN4(embedding)
                pred5 = self.FCN5(embedding)
                pred6 = self.FCN6(embedding)
                pred_list = [pred1, pred2, pred3, pred4, pred5, pred6]
                # pred_list=torch.cat([pred1,pred2,pred3,pred4,pred5,pred6],dim=1)
                return out, pred_list, embedding
                # return out,[pred1,pred2,pred3,pred4,pred5,pred6],embedding
        return out, pred, embedding

    def code(self, x):
        out = self.encoder(x)
        return out

    def explainableAELoss(self,y_true, y_pred):
            weight_decoder1 = self.decoder1.get_weights()
            weight_decoder2 = self.decoder1.get_weights()
            #ans = losses.binary_crossentropy(y_true, y_pred)#originally keras
            rate_site = 2.0
            rate_pathway = 2.0
            if self.toAddGeneSite:
                for i in range(len(weight_decoder2)):
                    print("weight[%d]*******************************************" % i)
                    # print(weight[i])
                    print(weight_decoder2[i].shape)
                print("self.gene_to_residue_info.gene_to_residue_map.shape")
                print(len(self.gene_to_residue_or_pathway_info.gene_to_residue_map))
                print(len(self.gene_to_residue_or_pathway_info.gene_to_residue_map[0]))

                regular_site = abs(
                    rate_site * weight_decoder2[15] * self.gene_to_residue_or_pathway_info.gene_to_residue_map_reversed)

                if self.toAddGenePathway:
                    regular_pathway = abs(
                        rate_pathway * weight_decoder1[12] * self.gene_to_residue_or_pathway_info.gene_pathway_reversed)
                    if self.lossMode == 'reg_mean':
                        ans += np.sum(regular_site) / len(regular_site) + np.sum(regular_pathway) / len(regular_pathway)
                    else:
                        ans += np.sum(regular_site) + np.sum(regular_pathway)
                elif self.lossMode == 'reg_mean':
                    ans += np.sum(regular_site) / len(regular_site)
                else:
                    ans += np.sum(regular_site)  # +1000*np.random.uniform(1)
            elif self.toAddGenePathway:
                regular_pathway = abs(
                    rate_pathway * weight_decoder2[12] * self.gene_to_residue_or_pathway_info.gene_pathway_reversed)
                if self.lossMode == 'reg_mean':
                    ans += np.sum(regular_pathway) / len(regular_pathway)
                else:
                    ans += np.sum(regular_pathway)
            return ans
    def save_site_gene_pathway_weight_visualization(self,info=""):
        import visualize_neural_network.VisualizeNN as VisNN
        #from sklearn.neural_network import MLPClassifier
        import numpy as np
        '''
        training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
        training_set_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T
        X = training_set_inputs
        y = training_set_outputs

        classifier = MLPClassifier(hidden_layer_sizes=(4,), alpha=0.01, tol=0.001, random_state=1)
        classifier.fit(X, y.ravel())
        '''

        network_structure = self.decoder1.parameters().shape#np.hstack(([X.shape[1]], np.asarray(classifier.hidden_layer_sizes), [y.shape[1]]))
        # Draw the Neural Network with weights
        network=VisNN.DrawNN(self.decoder1.parameters().shape, self.decoder1)
        network.draw(self.date+self.code+info+" decoder1 ")
        network=VisNN.DrawNN(self.decoder2.parameters().shape, self.decoder2)
        network.draw(self.date+self.code+info+" decoder2 ")

        # Draw the Neural Network without weights
        network=VisNN.DrawNN(self.decoder1.parameters().shape)
        network.draw(self.date+self.code+info+" decoder1 without weight")
        pass

  

class NN(nn.Module):
    def __init__(self, in_dim=784, h_dim=400):
        super(NN, self).__init__()
        mid_dim = int(math.sqrt(h_dim * in_dim))
        q1_dim = int(math.sqrt(h_dim * mid_dim))
        q3_dim = int(math.sqrt(mid_dim * in_dim))
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
            # nn.Linear(q1_dim, q3_dim),
            # nn.ReLU(),
            nn.Linear(q1_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Note: image dimension conversion will be handled by external methods
        """
        out = self.encoder(x)
        # out = self.decoder(out)
        return out

    def code(self, x):
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
