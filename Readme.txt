#Created by Wenjin Hu (E-mail  huwenjin@emails.bjut.edu.cn) from Beijing University of Technology @ 2021-9
#This is a demo on the CIFAR-10, NUS-WIDE, MS-COCO and ImageNet datasets with the BCMDH implementation based on the MatConvNet(http://www.vlfeat.org/matconvnet/) framework.
#Corresponding Author: Meng Jian (E-mail: jianmeng648@163.com, Website: https://scholar.google.com/citations?user=QSvCp7IAAAAJ&hl=en)
Paper：https://www.sciencedirect.com/science/article/abs/pii/S0925231221004793 

1. In the following is an example of using BCMDH on CIFAR-10.

    Stage 1.1:
    1. Download the CIFAR-10 dataset the website(https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz), unzip the file
       and put the folder 'data/CIFAR-10/.
    2. Download the Pretrained CNN model AlexNet from the website(http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat),
       and put it in the folder 'BCMDH/' .


    Stage 1.2:
    1. Installing and compiling the library in the MatConvNet, run 'matconvnet/compilenn.m' and 'matconvnet/setup.m'.
       Please refer to http://www.vlfeat.org/matconvnet/install/ for more information about installing MatConvNet.
    2. demo run command 'main_cifar10.m'.

2. In the following is an example of using BCMDH on NUS-WIDE, MS-COCO, and ImageNet datasets.
    Stage 2.1:
    1. Pre-processing NUS-WIDE , MS-COCO, and ImageNet datasets by yourself .
    2. Download the Pretrained CNN model AlexNet like CIFAR-10 demo.

    Stage 2.2:
    1. Installing and compiling the library in the MatConvNet like CIFAR-10 demo.
    2. demo run command 'main_nus.m' or 'main_coco.m' or 'main_imagenet.m'.
    3. Hint: please note that for NUS-WIDE dataset you should utilize the data splitting protocol in the paper, or you will lose some accuracy.
       That is, you should randomly sample 2100 query images (100 images per class) and 10500 training images (500 images per class) to construct query set and training set.


