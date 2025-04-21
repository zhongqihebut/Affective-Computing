# TSL
> [Temporal Sentiment Localization: Listen and Look in Untrimmed Videos](https://zzcheng.top/assets/pdf/2022_ACMMM_TSL300.pdf)

<!-- [ALGORITHM] -->
### Abstract

<p style="text-align: justify;">
Video sentiment analysis aims to uncover the underlying attitudes
of viewers, which has a wide range of applications in real world.
Existing works simply classify a video into a single sentimental
category, ignoring the fact that sentiment in untrimmed videos
may appear in multiple segments with varying lengths and unknown locations. To address this, we propose a challenging task,
i.e., Temporal Sentiment Localization (TSL), to find which parts
of the video convey sentiment. To systematically investigate fully and weakly-supervised settings for TSL, we first build a benchmark dataset named TSL-300, which is consisting of 300 videos
with a total length of 1,291 minutes. Each video is labeled in two
ways, one of which is frame-by-frame annotation for the fully supervised setting, and the other is single-frame annotation, i.e.,
only a single frame with strong sentiment is labeled per segment
for the weakly-supervised setting. Due to the high cost of labeling
a densely annotated dataset, we propose TSL-Net in this work, em ploying single-frame supervision to localize sentiment in videos. In
detail, we generate the pseudo labels for unlabeled frames using a
greedy search strategy, and fuse the affective features of both visual
and audio modalities to predict the temporal sentiment distribution.
Here, a reverse mapping strategy is designed for feature fusion, and
a contrastive loss is utilized to maintain the consistency between the
original feature and the reverse prediction. Extensive experiments show the superiority of our method against the state-of-the-art approaches.
</p>

## Data Preparation
1. Prepare [TSL-300](./assests/TSL-300_Data_Access_Form.docx) dataset.
    - We have provided constructed dataset and pre-extracted features.

2. Extract features with two-stream I3D networks
    - We recommend extracting features using [this repo](https://github.com/piergiaj/pytorch-i3d).
    - For convenience, we provide the features we used, which is also included in our dataset.
    - Link the features folder by using `sudo ln -s path-to-feature ./dataset/VideoSenti/`.
    
3. Place the features inside the `dataset` folder.
    - Please ensure the data structure is as below.

~~~~
├── dataset
   └── VideoSenti
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       ├── fps_dict.json
       ├── time.json
       ├── videosenti_gt.json
       ├── point_gaussian
           └── point_labels.csv
           ├── train
       └── features
           ├── train
               ├── rgb
                   ├── 1_Ekman6_disgust_3.npy
                   ├── 2_Ekman6_joy_1308.npy
                   └── ...
               └── logmfcc
                   ├── 1_Ekman6_disgust_3.npy
                   ├── 2_Ekman6_joy_1308.npy
                   └── ...
           └── test
               ├── rgb
                   ├── 9_CMU_MOSEI_lzVA--tIse0.npy
                   ├── 17_CMU_MOSEI_CbRexsp1HKw.npy
                   └── ...
               └── logmfcc
                   ├── 9_CMU_MOSEI_lzVA--tIse0.npy
                   ├── 17_CMU_MOSEI_CbRexsp1HKw.npy
                   └── ...
~~~~


### Training
```sh
python -W ignore ./main.py --model_path ./models/train --output_path ./outputs/train \
--log_path ./logs/train --seed 123
```
### Testing
```sh
python -W ignore ./main_eval.py --model_path ./models/test --output_path ./outputs/test \
--log_path ./logs/test --model_file ./models/train/model_seed_123.pth
```

The pre-trained model can be found in [pretrained model](https://drive.google.com/file/d/1IgtUszVJjoa-VJ_UZVryFFcMouVJ__7d/view?usp=sharing).

## Performance

| Metric                  | PyTorch  | Jittor  |
|-------------------------|----------|---------|
| Average Forward Time (train)    | 0.324s   | 0.068s  |
| Average Memory Usage (train)    | 11319MB  | 16132MB |
| Single Iteration Time(train)   | 1.162s   | 0.981s  |
| Average_mAP[0.1:0.3] (test)    | 0.1985   | 0.1949  |
| Average_pAP[0.1:0.3] (test)    | 0.2106   | 0.2095  |
| Average_nAP[0.1:0.3] (test)   | 0.1865   | 0.1803  |
| F2@AVG (test)                  | 0.3369   | 0.3577  |

## Citation

If you find this repo useful in your project or research, please consider citing the relevant publication.

````
@inproceedings{zhang2022temporal,
  title={Temporal Sentiment Localization: Listen and Look in Untrimmed Videos},
  author={Zhang, Zhicheng and Yang, Jufeng},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
````
````