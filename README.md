# PADNet

Peak area detection network (PADNet) directly learns to predict the phase region from the raw (2D) X-ray diffraction patterns (XRD) image without any need for explicit preprocessing and background removal. PADNet contains specially designed large symmetrical convolutional filters at the first layer to capture the peaks and automatically remove the background by computing the difference in intensity counts across different symmetries.

## Installation Requirements

Python 2.7

matplotlib 3.0.2

scipy 1.2.0

mpl_toolkits

Pillow 5.1.0

TensorFlow 1.12

scikit-learn 0.20.2

## Contents

This repository contains the code for preprocessing warped 2D X-ray diffraction patterns using minimum filter and convolutional smoothening, along with the code for training PADNet model on the XRD patterns.

* `bg_process.ipynb` - code for preprocessing warped 2D XRD images using minimum filter and convolutional smoothening.

* `load_data.py` - code for loading XRD images for training PADNet model.

* `model.py` - implementation of the architecture and training of PADNet models using different datasets.

* `perf_analysis.ipynb` - Jupyter notebook to train PADNet models using different cross validation ratios on different datasets.

* `train_utils.py` - utility code for training PADNet model.

* `training-data` - folder containing instructions on how to obtain the dataset used in the paper.


## Citation

Please cite the following work if you are using PADNet model and/or code for background preprocessing of 2D XRD patterns provided in this repository.

D. Jha, A. G. Kusne, R. Al-Bahrani, N. Nguyen, W.-keng Liao, A. Choudhary, and A. Agrawal, "Peak area detection network for directly learning phase regions from raw x-ray diffraction patterns." In 2019 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE. [DOI:10.1109/IJCNN.2019.8852096] [<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8852096">PDF</a>].


## Questions/Comments

email: dipendra.jha@eecs.northwestern.edu or ankitag@eecs.northwestern.edu</br>
Copyright (C) 2019, Northwestern University.<br/>
See COPYRIGHT notice in top-level directory.


## Funding Support

This work is supported in part by the following grants: NIST award 70NANB14H012, NSF award CCF-1409601; DOE awards DE-SC0014330, DE-SC0019358.
