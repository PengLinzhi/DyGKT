# DyGKT: Dynamic Graph for Adaptive Knowledge Tracing
This repository is built for the paper DyGKT: Dynamic Graph for Knowledge Tracing.

## Overview

We proposed a continuous dynamic graph neural network for knowledge tracing problems. Different from the existing work which extracts fixed-length question-answering sequences, we extend it to the infinite-length scale, making it more suitable for real-world scenarios.
Our experiment codes are written based on [DyGLib](https://github.com/yule-BUAA/DyGLib).


## Benchmark Datasets and Preprocessing

Five datasets are used in DyGLib, including [ASSITment12](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect), [ASSITment17](https://sites.google.com/view/assistmentsdatamining/dataset), [Slepemapy.cz](https://www.fi.muni.cz/adaptivelearning/?a=data), [junyi](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198), [EdNet-KT1](https://github.com/riiid/ednet).


## Knowledge Tracing Models

Eight classic or popular continuous-time knowledge tracing models are transformed into dynamic graph learning methods are included in our experiment, including models based on recurrent neural networks with the help of additional information or encoder, DKT, IEKT, LPKT, DIMKT, [CT-NCM](https://www.ijcai.org/proceedings/2022/0302.pdf), and QIKT. You can find most of the models in [pyKT](https://pykt-toolkit.readthedocs.io/en/latest/models.html)

## Dynamic Graph Models
Apart from traditional KT models, We conduct tests under general dynamic graph frameworks, TGN, TGAT, DyGFormer, and DyGKT to confirm the effectiveness of defining paradigms for training tasks under dynamic graphs and expressing dynamic features in the knowledge tracing task.

## Computation Cost Comparison

Unlike traditional KT approaches, we choose the dynamic graph structure rather than a sequential structure to model the learning process. Firstly, we sort all the question records in the dataset in ascending order of time. Each batch consists of a batch size of answering records, and each record contains a pair of nodes representing a student and a question. We only extract the historical interactions of these nodes for encoding and make predictions based on the presentation of the node pair. 

In contrast, traditional approaches and the three papers you mentioned consider a batch as a batch size of students, and they perform predictions on their question answering records as a whole. To facilitate training on these sequences, they often remove records that exceed 50/100 interactions, treating the sequence as static for training and prediction. However, we believe that such an approach cannot guarantee the dynamism of predicting student states and fails to fully capture the students' states without utilizing all of the data.

**Setting node feature dimension/hidden dimension as d, neighbor sample length as L, node memory dimension as M(M<d), number of edges as E, and number of students as S(S<E).**

Our time complexity is **O(2*EL)**, and spatial complexity is **O(Ed+NM)**. Because we need to predict each pair by the previous L records. Factor 2 is due to encoding calculations performed on both the neighbor sequences of the student node and the question node, as we don't present questions by embedding techniques.

<img width="325" alt="image" src="https://github.com/PengLinzhi/DyGKT/assets/73518557/cc999c5e-6d52-4f3d-8aab-e95a4f8822b5">


In the traditional static KT models, (L-1) predictions are made in the sequence models once the student's L interactions are put in. So the time complexity of the traditional static KT models is **O(SL)**, and the space complexity is **O(Ed+Nd)**. 

<img width="364" alt="image" src="https://github.com/PengLinzhi/DyGKT/assets/73518557/4400a8dd-9bf9-4584-acde-00e0a84960fc">

But we will also compare the traditional KT models when they are implemented within the dynamic graph. We expand the original KT methods to predict the pair of the student and the question based on the current student's historical answering sequence, and the sequence length maintains L. The time complexity for this calculation is **O(ELd)**, and the space complexity is **O(Ed+Nd)**. 

<img width="325" alt="image" src="https://github.com/PengLinzhi/DyGKT/assets/73518557/165b571f-9fe9-4798-8d42-968cfeb336c0">

The exact computation time of the model varies depending on the specific encoding method employed. For example, the AKT model utilizes attention mechanisms for computation, while the DKT model employs LSTM. The size of the model can measure the computational intensity. We have compiled a comparison of the methods used by all models in our paper, along with their respective time and space complexities, in Appendix Table 3.

## Evaluation Tasks
DyGLib supports dynamic link classification under both transductive and inductive settings with random negative sampling strategies.


## Incorporate or New Models
* For new models: Users can put the model implementation in  ```models``` folder, 
  and then create the model in ```train_xxx.py``` or ```evaluate_xxx.py``` to run the model.


## Environments

[PyTorch 1.8.1](https://pytorch.org/),
[numpy](https://github.com/numpy/numpy),
[pandas](https://github.com/pandas-dev/pandas),
[tqdm](https://github.com/tqdm/tqdm), and 
[tabulate](https://github.com/astanin/python-tabulate)


#### Model Training
* Example of training *DyGKT* on *assist17* dataset:
```{bash}
python train_link_classification.py --dataset_name assist17 --model_name DyGKT --num_neighbor 100 --num_runs 5 --gpu 0
```
#### Model Evaluation
* Example of evaluating *DyGKT* on *assist17* dataset:
```{bash}
python evaluate_link_classification.py --dataset_name assist17 --model_name DyGKT --num_neighbor 100 --num_runs 5 --gpu 0
```

## Acknowledgments

We are grateful to the authors of 
[DyGLib](https://github.com/yule-BUAA/DyGLib)
[pyKT](https://pykt-toolkit.readthedocs.io/en/latest/models.html) for making their project codes publicly available.


## Citation

