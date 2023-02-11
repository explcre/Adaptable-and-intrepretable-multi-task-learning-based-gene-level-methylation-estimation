# Adaptable and intrepretable Multi-task Learning based gene-level methylation estimation


## How to run the program
Use following command to run the program
```
python ./main.py
```

## Research Project Background
This project mainly focus on topics on methylation, which is a phenomenon in DNA which will cause dysfunction. We want use residual methylation data to predict the diseases.
However, the dimension of residual is enormous and the sample is comparatively fewer. Therefore, we want to propose a method to reduce the dimension and improve the performance.

We extract the information from residual methylation to get gene-level methylation, which is much lower in dimension.
Moreover, the gene-level methylation may give us some common and critical information which can be transferred among different datasets. We can use some represention learning method to extract the feature of the mechanism of gene-level methylation.



## The method we propose
First, we designed a refined auto-encoder architecture. Input is residual methylation and output is restored residual methylation. As we go deeper, the number of nodes in the layer becomes smaller in the first half and then increases to the same dimension as input. The two half parts are named encoder and decoder. For encoder, we can distill the inherently critical and low-dimension embedding data from enormous number of residuals without human labeling. We assume the bottleneck layer of the encoder represents pathway which provides information about the basic units of heredity. 

For decoder, we designed an explainable neural network which prunes the node. The network restore data from pathway to gene-level methylation, then to the residual methylation again. For each step, dimensions become larger and the former layer is a collection of the latter layer. The reason why it’s explainable is that we only remained the connection between certain residuals in the gene and certain genes in the pathway according to expert knowledge. This pruning method reduces the dimension to calculate and also can be explained by gene rules. 
Moreover, the auto-encoder can be adaptable in different datasets because the embedding can be shared between different input data sources, which can support transfer learning.
