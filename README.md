## Introduction

This is a project collection about affective computing  based on [Jittor](https://github.com/Jittor/jittor), created by the Computer Vision Laboratory at Nankai University.

This project contains three of our codes for affective computing, which are:TSL——《Temporal Sentiment Localization: Listen and Look in Untrimmed Videos》、CTEN——《Weakly Supervised Video Emotion Detection and Prediction via Cross-Modal Temporal Erasing Network》、BPM_GCN——《Looking into Gait for Perceiving Emotions via Bilateral Posture and Movement Graph Convolutional Networks》

## Dependencies

#### Recommended Environment

* Python 3.8.0
* jittor 1.3.9.14
* CUDA 11.3

## Requirements_jittor
* jittor==1.3.9.14
* PyYAML==6.0.2
* GPUtil==1.4.0
* fsspec==2023.6.0
* idna==3.4
* joblib==1.3.2
* numpy==1.22.0
* pandas==2.0.3
* Pillow==10.0.0
* protobuf==4.24.1
* regex==2023.8.8
* scikit-learn==1.3.0
* scipy==1.10.1
* tokenizers==0.13.3
* tqdm==4.66.1
* transformers==4.31.0
* triton==2.0.0
* networkx==3.1

**Step 1: Install the requirements**
```shell
git clone https://github.com/
cd 
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step 2: Install Affective Computing**

You can find specific information about the projects in the  folder ./project.

```shell
cd Affective Computing
cd projects
cd CTEN/TSL/BPM_GCN
```
After entering the folder of the specific project, you can learn  how to train the model and evaluate through the README.md.

### Datasets
CTEN——The used datasets are provided in [Ekman-6](https://github.com/kittenish/Frame-Transformer-Network), [VideoEmotion-8](https://drive.google.com/drive/folders/0B5peJ1MHnIWGd3pFbzMyTG5BSGs?resourcekey=0-hZ1jo5t1hIauRpYhYIvWYA), and [CAER](https://caer-dataset.github.io/).

TSL——For detailed information, please refer to the specific folder ./projects/TSL

BPM_GCN——For detailed information, please refer to the specific folder ./projects/BPM_GCN

## The Team
For more papers on sentiment analysis, feel free to follow  Vision and Learning Lab NKU  (https://cv.nankai.edu.cn/).

## Reference
1. [Jittor](https://github.com/Jittor/jittor)
2. [BPM_GCN](https://exped1230.github.io/BPM-GCN/GaitEmotion-BPM-GCN/index.html)
3. [CTEN](https://github.com/nku-zhichengzhang/WECL)
4. [TSL](https://github.com/nku-zhichengzhang/TSL300)
