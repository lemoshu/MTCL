# MTCL-HepaticVesselSeg2D
- Pytorch implementation for our Mean-Teacher-Assisted Confident Learning (MTCL) for hepatic vessel segmentation.
- Related paper was submitted to MICCAI 2021, crossed fingers!
- We will arrange the codes to release if the paper is accepted.

____
## Abstract
Manually segmenting the hepatic vessels from Computer Tomography (CT) is far more expertise-demanding and laborious than other structures due to the low-contrast and complex morphology of vessels, resulting in the extreme lack of high-quality labeled data. Without sufficient high-quality annotations, the usual data-driven learning-based approaches struggle with deficient training. On the other hand, directly introducing additional data with low-quality annotations may confuse the network, leading to undesirable performance degradation. To address this issue, we propose a novel mean-teacher-assisted confident learning framework to robustly exploit the noisy labeled data for the challenging hepatic vessel segmentation task. Specifically, with the adapted confident learning assisted by a third party, i.e., the weight-averaged teacher model, the noisy labels in the additional low-quality dataset can be transformed from ‘encumbrance’ to ‘treasure’ via progressive pixel-wise soft-correction, thus providing productive guidance.
____

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

## Usage

1. Clone the repo:
```
git clone https://https://github.com/HiLab-git/SSL4MIS.git 
cd SSL4MIS
```

2. Preprocess
- The preprocessing and file separating is a little complicated
- Available when paper is accepted - I will re-arrange the preprocess and training code after finishing the terribly defeated MS thesis (o_o)

3. Train the model
- Available when paper is accepted
```
cd code
python train_unet_2D_MT_IRCAD_concat_CL.py
```

4. Test the model
- Here the processed h5 files (concatenated volumes(img and prob map)) can be used for inference.  
- Download the testing data (10 cases, h5 files) and put the data in `../data/BraTS2019` or `../data/ACDC`.
```
cd code
python test_IRCAD_2D_c.py
```


