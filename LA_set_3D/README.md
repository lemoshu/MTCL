# MTCL-3D for Left Atrium Segmentation
3D version is available!


## Usage

1. Clone the repo:
```
cd MTCL/LA_set_3D
```

2. Data Preparation
Refer to ./data


3. Training script
```
cd ./code
python train_MTCL_3D.py
```

4. Test script 
```
cd ./code
python test_3D.py
```

## Citation
If our work brings some insights to you, please cite our paper as:
```
@article{xu2022anti,
  title={Anti-interference from Noisy Labels: Mean-Teacher-assisted Confident Learning for Medical Image Segmentation},
  author={Xu, Zhe and Lu, Donghuan and Luo, Jie and Wang, Yixin and Yan, Jiangpeng and Ma, Kai and Zheng, Yefeng and Tong, Raymond Kai-yu},
  journal={IEEE Transactions on Medical Imaging},
  year={2022},
  publisher={IEEE}
}

@artical{xu2021noisylabel,
  title={Noisy Labels are Treasure: Mean-Teacher-Assisted Confident Learning for Hepatic Vessel Segmentation},
  author={Zhe Xu, Donghuan Lu, Yixin Wang, Jie Luo, Jagadeesan Jayender, Kai Ma, Yefeng Zheng and Xiu Li},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention},
  year={2021}
}
```   
