# Adaptable and intrepretable Multi-task Learning based gene-level methylation estimation

## Introduction
-	Explored adaptable and interpretable neural network to find common genotype given 480k dimension sites, hundreds of sample. 
-	Designed an explainable site-gene-pathway ontology constraint to NN to discover new biomarkers by checking weights.
-	Implemented a Variational Auto-Encoder to support gene-level embedding shared among datasets to obtain multi-task learning.
-	Optimized a pretrain-finetune training scheme to increase accuracy by over 10%.

## Datasets
The method is tested on six datasets,including:

- Rheumatoid arthritis
- Systemic lupus erythematosus
- Multiple sclerosis
- Inflammatory bowel disease
- Psoriasis
- Type 1 diabetes

and is shown to have good performance in identifying common functions of DNA methylation in phenotypes.

## How to run the program
Use following command to install the prequisites:
```
pip install -r requirements.txt
```

Use following command to run the program:
```
python ./main.py
```

## Output of this program
After run `main.py`:

There will be some partial results in `./result/`

There will be a result for the setting of this run and all the test accuracy result in `./result-all/` named by the `output_file_name` in program `./tools.py` (around line 127). For example, `./result-all/1-10results-together.csv`.

There will be logs in `./log/` named by the date.

There will be cache file in `./cache/` if you will use same setting of number of residue for multiple dataset for multiple times, you can keep it so  that to save time for preprocessing.

There will be some data in `./tensorboard_log/`, you can use command 
```
tensorboard --logdir="./tensorboard_log/
``` 
to start the tensorboard to see the validation accuracy and weight distribution of each dataset each stages and each settings.

## Research Project Background
This project mainly focus on topics on methylation, which is a phenomenon in DNA which will cause dysfunction. We want use residual methylation data to predict the diseases.
However, the dimension of residual is enormous and the sample is comparatively fewer. Therefore, we want to propose a method to reduce the dimension and improve the performance.

We extract the information from residual methylation to get gene-level methylation, which is much lower in dimension.
Moreover, the gene-level methylation may give us some common and critical information which can be transferred among different datasets. We can use some represention learning method to extract the feature of the mechanism of gene-level methylation.



## The method we propose
First, we designed a refined auto-encoder architecture. Input is residual methylation and output is restored residual methylation. As we go deeper, the number of nodes in the layer becomes smaller in the first half and then increases to the same dimension as input. The two half parts are named encoder and decoder. For encoder, we can distill the inherently critical and low-dimension embedding data from enormous number of residuals without human labeling. We assume the bottleneck layer of the encoder represents pathway which provides information about the basic units of heredity. 

For decoder, we designed an explainable neural network which prunes the node. The network restore data from pathway to gene-level methylation, then to the residual methylation again. For each step, dimensions become larger and the former layer is a collection of the latter layer. The reason why it’s explainable is that we only remained the connection between certain residuals in the gene and certain genes in the pathway according to expert knowledge. This pruning method reduces the dimension to calculate and also can be explained by gene rules. 
Moreover, the auto-encoder can be adaptable in different datasets because the embedding can be shared between different input data sources, which can support transfer learning.
