# MTCL-HepaticVesselSeg2D
- This repo is to backup the Pytorch implementation for our MTCL for hepatic vessel segmentation.
- Related paper was submitted to MICCAI 2021, crossed fingers!
- We will arrange the codes to release if the paper is accepted.

____
## Abstract
Manually segmenting the hepatic vessels from Computer Tomography (CT) is far more expertise-demanding and laborious than other structures due to the low-contrast and complex morphology of vessels, resulting in the extreme lack of high-quality labeled data. Without sufficient high-quality annotations, the usual data-driven learning-based approaches struggle with deficient training. On the other hand, directly introducing additional data with low-quality annotations may confuse the network, leading to undesirable performance degradation. To address this issue, we propose a novel mean-teacher-assisted confident learning framework to robustly exploit the noisy labeled data for the challenging hepatic vessel segmentation task. Specifically, with the adapted confident learning assisted by a third party, i.e., the weight-averaged teacher model, the noisy labels in the additional low-quality dataset can be transformed from ‘encumbrance’ to ‘treasure’ via progressive pixel-wise soft-correction, thus providing productive guidance.
____

