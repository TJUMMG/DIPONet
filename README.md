## DIPONet: Dual-Information Progressive Optimization Network for Salient Object Detection.

### This is a PyTorch implementation of our proposed DIPONet for SOD. It has been accepted by DIGITAL SIGNAL PROCESSING.

## Prerequisites

- [Pytorch 0.4.1+](http://pytorch.org/)
- [torchvision](http://pytorch.org/)
- scipy 1.2.1
- opencv-python 4.4.0.44


## Usage

### 1. Clone the repository

```shell
git clone https://github.com/TJUMMG/DIPONet.git
cd DIPONet/
```

### 2. Download the datasets

Download the training dataset and unzip it into `./data` folder.
* [DUTS-TR] 

Download the testing datasets and unzip them into `./data` folder.
* [DUTS-TE] 
* [ECSSD] 
* [HKU-IS] 
* [PASCALS]   
* [SOD] 
* [DUTOMRON]  

For edge label, you can calculate them by using "get_edge_data.py". And you should put edge labels and sal labels together.
If you can't find these public SOD datasets, please concat us via: `liu_tju@tju.edu.cn`.

### 3. Download the pre-trained models for backbone (VGG16 and ResNet50)

Download the following pre-trained models [BaiDuYun]：https://pan.baidu.com/s/1qc6zgWf3aDAre6Ey_KQGlw 
提取码：4o44 into `./pretrained` folder. 

### 4. Train

1. Set the `image_root` and `gt_root` path in `train.py` correctly.

2. We demo using ResNet-50 and VGG-16 as network backbone and train with a initial lr of 2e-5 for 26 epoches, which is divided by 2 after 14 and 22 epochs. The input size is 448 * 448 and the batch size is 5.

3. We demo joint training with edge, you should put edge labels with saliency labels together.

4. After training the result model will be stored under `./trainresults/resnet` or `./trainresults/vgg` folder.

### 5. Test

1. Set the `test_root` and `dataset` and `model.load_state_dict(..)` path in `test.py` correctly.

2. The input size for testing is 448 * 448.

3. We demo joint training with edge, you can get edge results and saliency results.

4. After testing,  the result images will be stored under `./testresults/DIPONet_ResNet` or `./testresults/DIPONet_VGG` folder.


### 6. Pre-trained models, pre-computed results and evaluation tools

1. The DIPONet model trained by authors [BaiDuYun]：https://pan.baidu.com/s/1C-k2gepxcpPbHR4QEjbD6A  提取码：frf6 
2. we provide Saliency maps calculated by ourselves [BaiDuYun]: https://pan.baidu.com/s/1KLJxZzALrUflSj2NI-mcAg  提取码：0jdg 
3. All the evaluation results are calculated by using https://github.com/ArcherFMY/sal_eval_toolbox.

### 7. Contact
If you have any questions, feel free to contact us via: `liu_tju@tju.edu.cn` or `yuanmin@tju.edu.cn`.

