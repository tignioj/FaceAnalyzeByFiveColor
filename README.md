# Face analyze by five color

# 概览

## 1. 人脸检测

## 2. 人脸68个特征点提取

## 2. 肤色分割
### 1. 使用肤色阈值的几个融合算法[\[1\]](#reference), 代码在
test/skin_test/colorModel/hsv_rgb_ycrcb.py 
这东西跑了半天发现，它单独的YCrCb颜色空间跑起来效果相当糟糕，但是融合RGB，和HSV之后，效果还不错。


# Thanks 
https://github.com/sunyoe/FaceHealthDetect

# Reference
[1] [N.Rahman, K.Wei and J.See, RGB-H-CbCr Skin Colour Model for Human Face Detection(2006)Faculty of Information Technology, Multimedia University.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.718.1964&rep=rep1&type=pdf)