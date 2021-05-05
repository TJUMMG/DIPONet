## DIPONet: Dual-Information Progressive Optimization Network for Salient Object Detection.

### This is a PyTorch implementation of our proposed DIPONet for SOD. It has been submitted to IEEE Transactions on Circuits and Systems for Video Technology.

## Prerequisites

- [Pytorch 0.4.1+](http://pytorch.org/)
- [torchvision](http://pytorch.org/)
- scipy 1.2.1
- opencv-python 4.4.0.44

## Update

1. We released our code for joint training with edge, which is also our best performance model.
2. You may refer to this repo for results evaluation: [SalMetric](https://github.com/Andrew-Qibin/SalMetric).


## Usage

### 1. Clone the repository

```shell
git clone https://github.com/TJUMMG/DIPONet.git
cd DIPONet/
```

### 2. Download the datasets

Download the following datasets and unzip them into `data` folder.

* [MSRA-B and HKU-IS](https://drive.google.com/open?id=14RA-qr7JxU6iljLv6PbWUCQG0AJsEgmd) dataset. The .lst file for training is `data/msrab_hkuis/msrab_hkuis_train_no_small.lst`.
* [DUTS](https://drive.google.com/open?id=1immMDAPC9Eb2KCtGi6AdfvXvQJnSkHHo) dataset. The .lst file for training is `data/DUTS/DUTS-TR/train_pair.lst`.
* [BSDS-PASCAL](https://drive.google.com/open?id=1qx8eyDNAewAAc6hlYHx3B9LXvEGSIqQp) dataset. The .lst file for training is `./data/HED-BSDS_PASCAL/bsds_pascal_train_pair_r_val_r_small.lst`.
* [Datasets for testing](https://drive.google.com/open?id=1eB-59cMrYnhmMrz7hLWQ7mIssRaD-f4o).
* For edge label, you should calculate them by using "get_edge_data.py".

### 3. Download the pre-trained models for backbone (VGG16 and ResNet50)

Download the following pre-trained models [BaiDuYun]：https://pan.baidu.com/s/1qc6zgWf3aDAre6Ey_KQGlw 
提取码：4o44 into `./pretrained` folder. 

### 4. Train

1. Set the `image_root` and `gt_root` path in `train.py` correctly.

2. We demo using ResNet-50 and VGG-16 as network backbone and train with a initial lr of 2e-5 for 26 epoches, which is divided by 2 after 14 and 22 epochs. The input size is 448 * 448 and the batch size is 5.

3. We demo joint training with edge, you should calaclate edge labels by yourself and put them with saliency labels together.

4. After training the result model will be stored under `./trainresults/resnet` or `./trainresults/vgg` folder.

### 5. Test
The DIPONet model trained by authors [BaiDuYun]：https://pan.baidu.com/s/1at-NmppBhBy-T_hlgpowVw 提取码：pzfj 
aliency maps will be stored under `results/run-*-sal-*` folders in .png formats.


### 6. Pre-trained models, pre-computed results and evaluation tools

1. The DIPONet model trained by authors [BaiDuYun]：https://pan.baidu.com/s/1at-NmppBhBy-T_hlgpowVw  提取码：pzfj 
2. we provide Saliency maps calculated by ourselves [BaiDuYun]: https://pan.baidu.com/s/1KLJxZzALrUflSj2NI-mcAg  提取码：0jdg 
3. All the evaluation results are calculated by using https://github.com/ArcherFMY/sal_eval_toolbox.

### 7. Contact
If you have any questions, feel free to contact me via: `yuanmin@tju.edu.cn`.

