# CS-Net
<div align="center">
<img src="/image/csnet.png" height="640" width="400" >
<p>Overview of context splicing network.</p>
</div>

<div align="center">
<img src="/image/cs-block.png" height="480" width="480" >
<p>Overview of CS module.</p>
</div>

## Introduction
Automated volumetric medical image segmentation algorithms are highly demanded in 
clinical practice. With the advent of fully convolutional network (FCN) in semantic 
segmentation task, models based on FCN are proposed for medical image segmentation, 
and U-Net is one of the most successful model. However, U-Net and variations of U-Net 
always sacrifice feature resolution to pursue high-level features. Additionally, 
capacity of medical image segmentation models to capture multi-scale features and 
generate long-range dependencies are rarely concerned. In this paper, we exploit the 
latest image processing methods and propose a context splicing network (CS-Net) to 
extract rich contextual information while preserve spatial information. CS-Net 
consists of three major components: feature encoder module, context splicing 
module (CS module) and feature decoder module. The encoder is a pre-trained 
ResNet101 with atrous convolution. The CS module splice atrous spatial pyramid 
pooling (ASPP) block with a self-attention block that incorporates multi-scale 
features and global context. The proposed method is evaluated on three public 
datasets: [CHAOS](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/), [LiTS](https://competitions.codalab.org/competitions/17094) and [KiTS](https://kits19.grand-challenge.org), which are collected by computed tomography. 
Comprehensive results show that CS-Net outperforms the original U-Net and the 
recent proposed CE-Net for liver segmentation and kidney segmentation and has 
competitive tumor segmentation results.Some example results are showed below:

![Sample results of liver segmentation on CHAOS. 
From left to right: original images, ground truth, 
CS-Net, CE-Net and U-Net](/image/chaos-exam.png?raw=true)
Sample results of liver segmentation on CHAOS. 
From left to right: original images, ground truth, 
CS-Net, CE-Net and U-Net
![Sample results of liver segmentation on CHAOS. 
From left to right: original images, ground truth, 
CS-Net, CE-Net and U-Net](/image/lits-exam.png?raw=true)
Sample results of liver segmentation and tumor segmentation on LiTS. From left to right: original images, ground truth, CS-Net, CE-Net and U-Net
![Sample results of liver segmentation on CHAOS. 
From left to right: original images, ground truth, 
CS-Net, CE-Net and U-Net](/image/kits-exam.png?raw=true)
Sample results of kidney segmentation and tumor segmentation on KiTS. From left to right: original images, ground truth, CS-Net, CE-Net and U-Net
## Usage

In this repository, we provide the implementation of CS-Net which is based on nnUNet. It’s worth noting that we change the input from one channel to three channels. 
You can search `"dawn changed here"` globally in the project to see where we have modified.

And we change the model from the original nnUNet model to CS-Net in a rude way, you can see `generic_UNet.py` for details.

To set up CS-Net, follow the exactly same steps nnUNet given. Search `"set your dir"` to find where to set your directory quickly.

## Acknowledgment
This project is created based on [nnUnet](https://github.com/MIC-DKFZ/nnUNet), 
[deeplabv3+](https://github.com/jfzhang95/pytorch-deeplab-xception) 
and [DANet](https://github.com/junfu1115/DANet) and the authors retain all the copyright 
of the related codes. 

## Thanks to the reposities for study

### Data analysis

* [MICCAI-LITS2017](https://github.com/assassint2017/MICCAI-LITS2017)

### Data generator 
* [brats17](https://github.com/taigw/brats17)
* [H-DenseUNet](https://github.com/xmengli999/H-DenseUNet)
* [nnUnet](https://github.com/MIC-DKFZ/nnUNet)

### Models learning
* [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)
* [DANet](https://github.com/junfu1115/DANet)
* [CE-Net](https://github.com/Guzaiwang/CE-Net)
* [CCNet](https://github.com/speedinghzl/CCNet)
* [OCNet.pytorch](https://github.com/PkuRainBow/OCNet.pytorch)
* [PSANet_PyTorch](https://github.com/cfzd/PSANet_PyTorch)
* [Autofocus-Layer](https://github.com/yaq007/Autofocus-Layer)
