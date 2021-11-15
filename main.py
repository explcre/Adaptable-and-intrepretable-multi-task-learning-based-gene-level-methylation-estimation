import pandas as pd
from train import run
from predict import predict
from test import select_feature

code="GSE66695"
platform="platform.json"
model_type="VAE"
data_type="origin_data"

# train
train_data = pd.read_table("data_train.txt", index_col=0)
train_label = pd.read_table("label_train.txt", index_col=0).values.ravel()
latent_dim=5
run(code,train_data, train_label, platform, model_type, data_type,latent_dim)

# predict
test_data = pd.read_table("data_test.txt", index_col=0)
predict(code,test_data,platform,"GSE66695_LinearRegression_origin_datatrain_model.pickle",model_type, data_type)

# test(feature selection)
data = pd.read_table("GSE66695_gene_level(origin_data_LinearRegression).txt", index_col=0)
label = pd.read_table("label_test.txt", index_col=0).values.ravel()
select_feature(code,data,label,gene=True)

