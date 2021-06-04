# MTCL-HepaticVesselSeg2D
- Pytorch implementation for our Mean-Teacher-Assisted Confident Learning (MTCL) for hepatic vessel segmentation, which was early accepted by MICCAI'21.  
- We are still working on the extended version of this paper, and will re-organize the codes after that.

____
## Abstract
Manually segmenting the hepatic vessels from Computer Tomography (CT) is far more expertise-demanding and laborious than other structures due to the low-contrast and complex morphology of vessels, resulting in the extreme lack of high-quality labeled data. Without sufficient high-quality annotations, the usual data-driven learning-based approaches struggle with deficient training. On the other hand, directly introducing additional data with low-quality annotations may confuse the network, leading to undesirable performance degradation. To address this issue, we propose a novel mean-teacher-assisted confident learning framework to robustly exploit the noisy labeled data for the challenging hepatic vessel segmentation task. Specifically, with the adapted confident learning assisted by a third party, i.e., the weight-averaged teacher model, the noisy labels in the additional low-quality dataset can be transformed from ‘encumbrance’ to ‘treasure’ via progressive pixel-wise soft-correction, thus providing productive guidance.
____

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* Python == 3.6 
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

## Usage

1. Clone the repo:
```
cd MTCL
```

Install the environment
```
pip install -r XX.txt
```

2. Dataset acquisition and Preprocess scripts
- For the dataset, please refer to [3DIRCADb](https://www.ircad.fr/research/3d-ircadb-01/) and [Medical Segmentation Decathlon (Task 8)](http://medicaldecathlon.com/). Note that we combine the masks of portalvein and venacava for IRCADb dataset, and the liver masks of MSD8 are obtained from the publicly available trained [H-DenseUNet model](https://github.com/xmengli999/H-DenseUNet). Thanks for their nice work.  

- After acquiring the datasets, you can refer to the following preprocessing scripts. The preprocessing undergoes masking, ROI cropping, normalization and Sato-based vessel prob map generation. In practice, we processed the data into h5 format.  
```
dataloaders/
├── 1_ROI_preprocess.py                       > Generate processed hepatic CT image for IRCADb                   
├── 1_ROI_preprocess_MSD.py                   > Generate processed hepatic CT image for MSD8 
├── 1_VesselEnhance.py                        > Generate Sato Vessel Prob Map for IRCADb 
├── 1_VesselEnhance_MSD.py                    > Generate Sato Vessel Prob Map for MSD8 
├── 2_IRCAD_data_processing.py                > Convert processed CT img to h5 file                   
├── 2_IRCAD_Prob_concat.py                    > concatenate the processed img and Sato Prob Map, and convert to h5 file  
├── 2_MSD_data_processing.py                  > Convert processed CT img to h5 file (MSD8)                   
├── 2_MSD_Prob_concat.py                      > concatenate the processed img and Sato Prob Map, and convert to h5 file (MSD8) 
├── 3_NEW_file_seperate.py                    > file list generate (txt) 
├── allinone_inference_preprocess.py          > all in one w/o txt list generator 
├── dataset.py                                > functions for dataloaders in pytorch
└── utils

```


3. Training script
```
cd code
python train_unet_2D_MT_IRCAD_concat_CL.py
```

4. Test script
- The processed h5 files (concatenated volumes (img and prob map)) should be used for inference. Pls refer to the above preprocessing script.    
```
cd code
python test_IRCAD_2D_c.py
```

## Citation
If our work brings some insights to you, please cite our paper as:
```
@artical{xu2021noisylabel,
  title={Noisy Labels are Treasure: Mean-Teacher-Assisted Confident Learning for Hepatic Vessel Segmentation},
  author={Zhe Xu, Donghuan Lu, Yixin Wang, Jie Luo, Jagadeesan Jayender, Kai Ma, Yefeng Zheng and Xiu Li},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2021}
}
```

## Acknowledgement
The training and test scripts are partly based on the project [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).  
