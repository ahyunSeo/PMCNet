## Learning to Discover Reflection Symmetry via Polar Matching Convolution
<p align="center">
Ahyun Seo*, Woohyeon Shim*, Minsu Cho
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2108.12952">[paper]</a>
    <a href="http://cvlab.postech.ac.kr/research/PMCNet">[project page]</a>
</p>

<!-- [[paper]](https://arxiv.org/abs/2108.12952) [[project page]](http://cvlab.postech.ac.kr/research/PMCNet/) -->

Official PyTorch implementation of *Learning to Discover Reflection Symmetry via Polar Matching Convolution (ICCV 2021)*.

Contributors of this repo: [Woohyeon Shim](https://github.com/shim94kr), [Ahyun Seo](https://github.com/ahyunSeo)

### Environment
```
    conda create --name pmcnet python=3.7
    conda activate pmcnet
    conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=11.0 -c pytorch
    conda install -c conda-forge matplotlib
    pip install albumentations tqdm parmap scikit-image pycocotools opencv-python
    
    mkdir weights
    # setup coco_path and sym_datasets
    cd bsds
    python setup.py build_ext --inplace

```

### Datasets
- symmetry detection datasets (LDRS, SDRW, NYU) (passwd: ldrs2021)
[sym_datasets](https://postechackr-my.sharepoint.com/:u:/g/personal/lastborn94_postech_ac_kr/EQdRWpc9HiRDqgdQohA3X-oBuoeUS6d8U24dRykhsL1vnw?e=eQ2vaN)
- COCO dataset (2014)
[train2014](http://images.cocodataset.org/zips/train2014.zip)
[val2014](http://images.cocodataset.org/zips/val2014.zip)
[annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

```
.
├── coco_path
│   ├── train2014
│   ├── val2014
│   └── annotations
├── sym_datasets
│   ├── NYU
│   ├── SDRW
│   └── LDRS
├── (...) 
└── main.py
```

### Training
The trained weights and arguments will be save to the checkpoint path corresponding to the VERSION_NAME.

```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --ver VERSION_NAME 
```

### Test
- trained weights of PMCNet (passwd: ldrs2021)
[weights](https://postechackr-my.sharepoint.com/:u:/g/personal/lastborn94_postech_ac_kr/EeWlTzf8JhlCtK2cPoT7WQYB10sHsq7g-OxOQnvcXlgb2A?e=KsxEye) 

```
    CUDA_VISIBLE_DEVICES=0 python main.py --ver ours -t
```

### References

- Python port of BSDS 500 [link](https://github.com/Britefury/py-bsds500)
- spb-mil [link](https://github.com/tsogkas/spb-mil)
- NYU database [link](https://symmetry.cs.nyu.edu/)
- SDRW dataset (CVPR 2013 challenge) [link](http://vision.cse.psu.edu/research/symComp13/index.shtml)
- COCO preprocessing [link](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/data/dataloader/mscoco.py)



### Citation
If you find our code or paper useful to your research work, please consider citing:
```
@inproceedings{seoshim2021pmcnet,
    author   = {Seo, Ahyun and Shim, Woohyeon and Cho, Minsu},
    title    = {Learning to Discover Reflection Symmetry via Polar Matching Convolution},
    booktitle= {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year     = {2021}
}
```
