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


class MeiNN(nn.Module):
    def __init__(self, config, path, date, code, X_train, y_train, platform, model_type, data_type,
                 HIDDEN_DIMENSION, toTrainMeiNN, AE_epoch_from_main=1000, NN_epoch_from_main=1000,
                 separatelyTrainAE_NN=True,
                 model_dir='./saved_model/',
                 train_dataset_filename=r"./dataset/data_train.txt", train_label_filename=r"./dataset/label_train.txt",
                 gene_to_site_dir=r"./platform.json", gene_to_residue_or_pathway_info=None,
                 toAddGeneSite=True, toAddGenePathway=True,
                 multiDatasetMode="softmax", datasetNameList=[], lossMode='reg_mean'):
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
        out_dim = len(self.datasetNameList)

        mid_dim = int(math.sqrt(in_dim * out_dim))
        q3_dim = int(math.sqrt(in_dim * mid_dim))

        q1_dim = int(math.sqrt(latent_dim * mid_dim))
        self.FCN = nn.Sequential(
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


    def forward(self, x):
        """
        Note: image dimension conversion will be handled by external methods
        """
        embedding = self.encoder(x)
        out = self.decoder(embedding)
        pred = self.FCN(embedding)
        if self.multiDatasetMode=='multi-task':
            pred1= self.FCN(embedding)
            pred2 = self.FCN(embedding)
            pred3 = self.FCN(embedding)
            pred4 = self.FCN(embedding)
            pred5 = self.FCN(embedding)
            pred6 = self.FCN(embedding)
            return out,[pred1,pred2,pred3,pred4,pred5,pred6],embedding

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
