import pandas as pd
from train import run
from predict import predict
from test import select_feature

code = "GSE66695"
platform = "platform.json"
model_type = "AE"#"RandomForest"
predict_model_type="L2"
data_type = "origin_data"
dataset_type="train"
isTrain=True
toTrainAE=True
toTrainNN=True
model=None
# train
if isTrain:
    train_data = pd.read_table("data_train.txt", index_col=0)
    train_label = pd.read_table("label_train.txt", index_col=0).values.ravel()
    model=run(code, train_data, train_label, platform, model_type, data_type,79*5,toTrainAE,toTrainNN)


# predict
test_data = pd.read_table("data_test.txt", index_col=0)
predict(code, test_data, platform, code +"_"+model_type+"_"+data_type+dataset_type+"_model.pickle", model_type, data_type,model,predict_model_type)

# test(feature selection)
data = pd.read_table(code+"_gene_level("+data_type+"_"+model_type+").txt", index_col=0)
label = pd.read_table("label_test.txt", index_col=0).values.ravel()
select_feature(code, data, label, gene=True)
