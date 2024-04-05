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

## Evaluation Tasks

DyGLib supports dynamic link classification under both transductive and inductive settings with random negative sampling strategies.


## Incorporate New Datasets or New Models

New datasets and new models are welcomed to be incorporated into DyGLib by pull requests.
* For new datasets: The format of new datasets should satisfy the requirements in DyGLib. 
  Users can put the new datasets in ```DG_data``` folder, and then run ```preprocess_data/preprocess_data.py``` to get the processed datasets.
  
* For new models: Users can put the model implementation in  ```models``` folder, 
  and then create the model in ```train_xxx.py``` or ```evaluate_xxx.py``` to run the model.


## Environments

[PyTorch 1.8.1](https://pytorch.org/),
[numpy](https://github.com/numpy/numpy),
[pandas](https://github.com/pandas-dev/pandas),
[tqdm](https://github.com/tqdm/tqdm), and 
[tabulate](https://github.com/astanin/python-tabulate)

#### Data Preparing
See detailed information in [DyGLib](https://github.com/yule-BUAA/DyGLib)

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

