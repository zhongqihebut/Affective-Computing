# BPM_GCN
> [Looking into Gait for Perceiving Emotions via Bilateral Posture and Movement Graph Convolutional Networks](https://exped1230.github.io/BPM-GCN/GaitEmotion-BPM-GCN/static/pdfs/TAFFC_GaitEmotion_BPM_GCN.pdf)

<br>
[[Project Page]](https://exped1230.github.io/BPM-GCN/GaitEmotion-BPM-GCN/index.html)</br>

<!-- [ALGORITHM] -->
### Abstract

<p style="text-align: justify;">
Emotions can be perceived from a person's gait, i.e., their walking style. Existing methods on gait emotion recognition mainly leverage the posture information as input,
but ignore the body movement, which contains complementary information for recognizing emotions evoked in the gait. In this paper, we propose a Bilateral Posture and
Movement Graph Convolutional Network (BPM-GCN) that consists of two parallel streams, namely posture stream and movement stream, to recognize emotions from
two views. The posture stream aims to explicitly analyse the emotional state of the person. Specifically, we design a novel regression constraint based on the hand-
engineered features to distill the prior affective knowledge into the network and boost the representation learning. The movement stream is designed to describe the
intensity of the emotion, which is an implicitly cue for recognizing emotions. To achieve this goal, we employ a higher-order velocity-acceleration pair to construct
graphs, in which the informative movement features are utilized. Besides, we design a PM-Interacted feature fusion mechanism to adaptively integrate the features from
the two streams. Therefore, the two streams collaboratively contribute to the performance from two complementary views. Extensive experiments on the largest
benchmark dataset Emotion-Gait show that BPM-GCN performs favorably against the state-of-the-art approaches (with at least 4.59% performance improvement).
</p>

## Dataset
 The used datasets are provided in the homepage of [Emotion-Gait](https://gamma.umd.edu/software). 

Note that since the dataset is changed on the official website, we provide the original dataset. The code and dataset are provided for research only.
[Baidu Drive](https://pan.baidu.com/s/1rNC7SQrwNnZBVMRzPaifZA) 
(acil)

## Training
```
python main_diff_combine_double_fagg.py
```


The pre-trained model can be found in [pretrained model](https://pan.baidu.com/s/1Rzc-j16BNOsbh9V1FiZdcg?pwd=btef).

## Performance

| Metric             | PyTorch | Jittor  |
|--------------------|---------|---------|
| Average Forward Time (train) | 0.218s  | 0.054s  |
| Average Memory Usage (train) | 3599MB  | 6899MB  |
| Single epoch Time(train) | 17.318s | 15.790s |
| Average_Acc(test)  | 0.8899  | 0.9115  |
| Happy_Acc(test)  | 0.9683  | 0.9820  |
| Sad_Acc(test)  | 0.8049  | 0.9756  |
| Angry_Acc(test)  | 0.7879  | 0.7778  |
| Neutral_Acc(test)  | 0.7222  | 0.3846  |

## Citation
If you find this repo useful in your project or research, please consider citing the relevant publication.

**Bibtex Citation**
````
@article{zhai2024Looking,
  author={Zhai, Yingjie and Jia, Guoli and Lai, Yu-Kun and Zhang, Jing and Yang, Jufeng and Tao, Dacheng}
  journal={IEEE Transactions on Affective Computing}, 
  title={Looking into Gait for Perceiving Emotions via Bilateral Posture and Movement Graph Convolutional Networks}, 
  year={2024}
}

````