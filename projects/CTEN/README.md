# TSL
> [Weakly Supervised Video Emotion Detection and Prediction via Cross-Modal Temporal Erasing Network](https://github.com/nku-zhichengzhang/CTEN/blob/main/assests/cvpr23_WECL.pdf)

<!-- [ALGORITHM] -->
### Abstract

<p style="text-align: justify;">
Automatically predicting the emotions of user-generated
videos (UGVs) receives increasing interest recently. How-
ever, existing methods mainly focus on a few key visual
frames, which may limit their capacity to encode the con-
text that depicts the intended emotions. To tackle that,
in this paper, we propose a cross-modal temporal eras-
ing network that locates not only keyframes but also con-
text and audio-related information in a weakly-supervised
manner. In specific, we first leverage the intra- and
inter-modal relationship among different segments to ac-
curately select keyframes. Then, we iteratively erase
keyframes to encourage the model to concentrate on the
contexts that include complementary information. Exten-
sive experiments on three challenging video emotion bench-
marks demonstrate that our method performs favorably
against state-of-the-art approaches.
</p>
### Training&Testing
```sh
python main_erase.py
```
The pre-trained model can be found in [pretrained model]( https://pan.baidu.com/s/1RcLG4CJNgkAs9-MEoxtgnQ?pwd=7gst).


## Performance


| Metric             | PyTorch | Jittor  |
|--------------------|---------|---------|
| Average Forward Time (train) | 0.22s  | 0.13s  |
| Average Memory Usage (train) | 12380MB  | 15673MB  |
| Single epoch Time(train) | 172s | 468s |
| Average_Acc(test)  | 49.1%  | 26.9%  |

## Citation
If you find this repo useful in your project or research, please consider citing the relevant publication.

**Bibtex Citation**
````
@InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Zhicheng and Wang, Lijuan and Yang, Jufeng},
    title     = {Weakly Supervised Video Emotion Detection and Prediction via Cross-Modal Temporal Erasing Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {18888-18897}
}
````