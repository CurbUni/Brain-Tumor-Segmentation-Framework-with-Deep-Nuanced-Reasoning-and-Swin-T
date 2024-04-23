# Brain-Tumor-Segmentation-Framework-with-Deep-Nuanced-Reasoning-and-Swin-T

## 相关仓库

| 模型             | 路径                                                         |
| ---------------- | ------------------------------------------------------------ |
| U-net            | https://github.com/bubbliiiing/unet-pytorch                  |
| Unet 3+          | https://github.com/ZJUGiveLab/UNet-Version                   |
| swin transformer | https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation |

### 所需环境

安装requirements.txt依赖

pip install -r requirements.txt

### 训练步骤

数据集准备

#### 数据集下载

https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

#### 数据集预处理

运行data_progress.py

#### 训练数据集

运行train.py

```python
CUDA_VISIBLE_DEVICES=0，1,2,3，4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 20003 train.py
```





