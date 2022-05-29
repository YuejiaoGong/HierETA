HierETA
---------------

This repository is the implementation of our KDD'22 Applied Data Science Track paper:

>Interpreting Trajectories from Multiple Views: A Hierarchical Self-Attention Network for Estimating the Time of Arrival.  Zebin Chen, Xiaolin Xiao, Yue-Jiao Gong, Jun Fang, Nan Ma, Hua Chai, Zhiguang Cao. KDD 2022.


## Required packages

The code has been tested running under Python 3.8.5, with the following packages installed (along with their dependencies):

- numpy==1.19.2
- scipy==1.6.2
- torch==1.8.0
- tensorboardX==2.2


## Files in the folder
Here we provide the source code and part of desensitized sample data. You can replace the samples with your own data easily.

The folder is organised as follows:
- `data-info/` contains:
    - `data_info.json` is the statistical information of different route attributes.
    - `segment_attrs.json` mapps segID to segment attributes, such as length, functional_level, lane number, et.al.
- `models/` contains the implementation of HierETA network.
- `samples/` contains some desensitized data, each row represents a unique travel order.
- `dataloading.py` contains tools for loading dataset.
- `log.py` manages log write.
- `main.py` provides full training/testing run on the dataset.
- `utils.py` contains tools for metric calculation.


## How to Run
```shell
python main.py
```
You can perform training/testing or parameter tuning by adjusting the `ArgumentParser's` options. Please refer to `main.py` for details.


## Citations

```bibtex
@inproceedings{chen2022hiereta,
    title     = {Interpreting Trajectories from Multiple Views: A Hierarchical Self-Attention Network for Estimating the Time of Arrival},
    author    = {Chen, Zebin and Xiao, Xiaolin and Gong, Yue-Jiao and Fang, Jun and Ma, Nan and Chai, Hua and Cao, Zhiguang},
    booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
    year      = {2022}
```