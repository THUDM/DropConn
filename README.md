# DropConn: Dropout Connection Based Random GNNs for Molecular Property Prediction

DropConn is an adaptive data augmentation strategy, which better leverages edge features and assigns weights for chemical bonds to emphasize their importance, and generates robust representations. To our best knowledge, DropConn is the first work generating automated data by mining information from existing edge features for molecule graph property prediction.

This is our PyTorch implementation for TKDE'2023 paper.

## Environment Requirements

The code has been tested running under Python 3.9.7. The required packages are as follows:

- pytorch == 1.9.1+cu111

## Citing

If you find this work is helpful to your research, please consider citing our paper:

```
@ARTICLE{10164235,
  author={Zhang, Dan and Feng, Wenzheng and Wang, Yuandong and Qi, Zhongang and Shan, Ying and Tang, Jie},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={DropConn: Dropout Connection Based Random GNNs for Molecular Property Prediction}, 
  year={2023},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TKDE.2023.3290032}}
```
