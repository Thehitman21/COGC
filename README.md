# COGC
A Pytorch Implementation of COGC

Controllable Optimizable Gamma Correction as Efficient Image Luminance Adapter. In BMVC'2026 <br>
[Paper]()

<img src="figs/results.png" width="800px"/>

### Requirements ###
1. Python
2. Torch >= 1.5.0
3. numpy, PIL, BM3D

### Testing  Usage ###
To quickly test your own images with our model, you can just run through
```shell
python main.py 
    --use_gpu=1 \                           # use gpu or not
    --gpu_idx=0 \
    --gpu_mem=0.5 \                         # gpu memory usage
    --phase=test \
    --test_dir=/path/to/your/test/dir/ \
    --save_dir=/path/to/save/results/ \
    --decom=0                               # save only enhanced results or together with decomposition results
```

 ### Citation ###
 ```
 @inproceedings{Yang2026COGC,
  title={Controllable Optimizable Gamma Correction as Efficient Image Luminance Adapter},
  author={Yimu Yang, Jiaming Chen, Zhenzhong Feng, Zhenyuan Yang},
  booktitle={British Machine Vision Conference},
  year={2026},
  organization={British Machine Vision Association}
}
```
