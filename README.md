# Brain-Tumor-Segmentation-Framework-with-Deep-Nuanced-Reasoning-and-Swin-T

## Related warehouses

| 模型             | 路径                                                         |
| ---------------- | ------------------------------------------------------------ |
| U-net            | https://github.com/bubbliiiing/unet-pytorch                  |
| Unet 3+          | https://github.com/ZJUGiveLab/UNet-Version                   |
| swin transformer | https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation |

## Required environment

Install the requirements.txt dependencies

```
pip install -r requirements.txt
```

## Training steps

Dataset preparation

#### Download dataset

https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

#### Dataset preprocessing

Running the data_progress.py

```python
python data_progress.py
```

#### Model training

Running the train.py

```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 20003 train.py
```



### Citing Brain-Tumor-Segmentation-Framework-with-Deep-Nuanced-Reasoning-and-Swin-T.

```
This will be presented after the paper is published.
```

