<h2 align="left">KS-DETR: Knowledge Sharing in Attention Learning for Detection Transformer</h2>


We release our code for our submitted manuscript `KS-DETR: Knowledge Sharing in Attention Learning for Detection Transformer`.

[comment]: <> ([comment]: <> &#40;Solarized dark             |  Solarized Ocean&#41;)

[comment]: <> (:-------------------------:|:-------------------------:)

[comment]: <> (![ks-detr arch]&#40;./projects/ks_detr/assets/ks-detr-freamework.png&#41;  |  ![triple attention]&#40;./projects/ks_detr/assets/triple_attention.png&#41;)


<div align="center">

  <img src="./projects/ks_detr/assets/ks-detr-freamework.png" width="45%"/>

 <img src="./projects/ks_detr/assets/triple_attention.png" width="45%"/>
</div><br/>

[comment]: <> (  <img src="./projects/ks_detr/assets/teacher-attn-accu.png"/>)
 

## Main results and Pretrained Models


Here we provide the pretrained `KS-DETR` weights based on detrex.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Pretrain</th>
<th valign="bottom">Epochs</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<!-- ROW: ks_dab_detr_r50_50ep -->
<tr><td align="left"><a href="projects/ks_detr/configs/ks_dab_detr/ks_dab_detr_r50_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-DAB-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">43.9</td>
<td align="center"> <a href="https://drive.google.com/file/d/1TjIGuNlrUg1u2oFkiULpYIZmGqz-ZszC/view?usp=share_link">model</a></td>
</tr>


<!-- ROW: ks_dab_detr_r101_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_dab_detr/ks_dab_detr_r101_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-DAB-DETR-R101</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">45.3</td>
<td align="center"> <a href="https://drive.google.com/file/d/1RM44UQsXsvq6_7_2UDnspRlQVAnZhmWJ/view?usp=share_link">model</a></td>

<!-- ROW: ks_dab_detr_swin_t_in1k_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_dab_detr/ks_dab_detr_swin_tiny_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-DAB-DETR-Swin-T</a></td>
<td align="center">Swin-T</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">47.1</td>
<td align="center"> <a href="https://drive.google.com/file/d/1P971eidVaB0nt9Uhs6tOG79-q98G9eHg/view?usp=share_link">model</a></td>
</tr>




<!-- ROW: ks_conditional_detr_r50 -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_conditional_detr/ks_conditional_detr_r50_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-Conditional-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">45.3</td>
<td align="center"> <a href="https://drive.google.com/file/d/1P971eidVaB0nt9Uhs6tOG79-q98G9eHg/view?usp=share_link">model</a></td>

<!-- ROW: ks_conditional_detr_r101_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_conditional_detr/ks_conditional_detr_r101_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-Conditional-DETR-R101</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">47.1</td>
<td align="center"> <a href="https://drive.google.com/file/d/1amZlGUnBBkrn-HIySklI7A0yB3erDrxk/view?usp=share_link">model</a></td>
</tr>


<!-- ROW: ks_dn_detr_r50_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_dn_detr/ks_dn_detr_r50_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-DN-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">45.2</td>
<td align="center"> <a href="https://drive.google.com/file/d/1CemLoPayF52HYFEydDJ-B5F-_BwBeptC/view?usp=share_link">model</a></td>

<!-- ROW: ks_dn_detr_r101_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_dn_detr/ks_dn_detr_r101_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-DN-DETR-R101</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">46.5</td>
<td align="center"> <a href="https://drive.google.com/file/d/1fd2qTXZnGxocq5m5rc97uFRa0WbhiSJl/view?usp=share_link">model</a></td>
</tr>



<!-- ROW:  -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_deformable_detr/ks_deformable_detr_r50_12ep_smlp_qkv_triple_attn_outproj_ffn_v0.py">
KS-Deformable-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">36.4</td>
<td align="center"> <a href="https://drive.google.com/file/d/1tYM_c_Q2j3LEY_EZ_hTJa3XvT0oxaFu9/view?usp=share_link">model</a></td>

<!-- ROW:  -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_deformable_detr/ks_deformable_detr_r101_12ep_smlp_qkv_triple_attn_outproj_ffn_v0.py">
KS-Deformable-DETR-R101</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">38.4</td>
<td align="center"> <a href="https://drive.google.com/file/d/1fd2qTXZnGxocq5m5rc97uFRa0WbhiSJl/view?usp=share_link">model</a></td>
</tr>

 <tr><td align="left"><a href="projects/ks_detr/configs/ks_dn_deformable_detr/ks_dn_deformable_detr_r50_12ep_smlp_qkv_triple_attn_outproj_ffn_v0.py">
KS-DN-Deformable-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">46.5</td>
<td align="center"> <a href="https://drive.google.com/file/d/1fd2qTXZnGxocq5m5rc97uFRa0WbhiSJl/view?usp=share_link">model</a></td>

<!-- ROW: ks_deformable_detr_r50_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_deformable_detr/ks_deformable_detr_r50_50ep_smlp_qkv_triple_attn_outproj_ffn_v0.py">
KS-Deformable-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">44.8</td>
<td align="center"> <a href="https://drive.google.com/file/d/1fd2qTXZnGxocq5m5rc97uFRa0WbhiSJl/view?usp=share_link">model</a></td>


<!-- ROW: ks_deformable_detr_r101_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_deformable_detr/ks_deformable_detr_r101_50ep_smlp_qkv_triple_attn_outproj_ffn_v0.py">
KS-Deformable-DETR-R101</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">46.0</td>
<td align="center"> <a href="https://drive.google.com/file/d/1fd2qTXZnGxocq5m5rc97uFRa0WbhiSJl/view?usp=share_link">model</a></td>
</tr>

</tbody></table>



[comment]: <> (## What's New)



## Installation

```shell

conda create -n ksdetr python=3.8 -y
conda activate ksdetr

git clone https://github.com/edocanonymous/KS-DETR
cd KS-DETR
python -m pip install -e detectron2
pip install -e .

```

[comment]: <> (## Getting Started)



## Training
To train the models with R101 backbone, the pretrained `IN1k` weights should be available at location `output/weights/R-101.pkl`.
We can follow  [`https://github.com/facebookresearch/detectron2/blob/main/tools/convert-torchvision-to-d2.py`](https://github.com/facebookresearch/detectron2/blob/main/tools/convert-torchvision-to-d2.py)
to convert [`https://download.pytorch.org/models/resnet101-5d3b4d8f.pth`](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) 
to torchvision format and obtain `R-101.pkl` by 

```shell
 wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth -O output/r101.pth
 python ./detectron2/tools/convert-torchvision-to-d2.py output/r101.pth output/weights/R-101.pkl
```
We provide our converted `R-101.pkl` file [here](https://drive.google.com/file/d/1OpXH1hlLI87ochfpOhU__Oohni6VEgeV/view?usp=share_link).

All configs can be trained with:
```bash
cd detrex
python tools/train_net.py --config-file projects/dab_detr/configs/path/to/config.py --num-gpus 8
```



To train `KS-DAB-DETR-R50`, `KS-DAB-DETR-R101`, and `KS-DAB-DETR-Swin-T`,
```bash
python tools/train_net.py --config-file projects/ks_detr/configs/ks_dab_detr/ks_dab_detr_r50_50ep_smlp_qkv_triple_attn.py --num-gpus 8

python tools/train_net.py --config-file projects/ks_detr/configs/ks_dab_detr/ks_dab_detr_r101_50ep_smlp_qkv_triple_attn.py --num-gpus 8

python tools/train_net.py --config-file projects/ks_detr/configs/ks_dab_detr/ks_dab_detr_swin_tiny_50ep_smlp_qkv_triple_attn.py --num-gpus 8

```


## Evaluation
Model evaluation can be done as follows:
```bash
cd detrex
python tools/train_net.py --config-file projects/dab_detr/configs/path/to/config.py --eval-only train.init_checkpoint=/path/to/model_checkpoint
```

## License

This project is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement
- Our code is built on detrex, which is an open-source toolbox for Transformer-based detection algorithms created by researchers of **IDEACVR**. 

- detrex is built based on [Detectron2](https://github.com/facebookresearch/detectron2) and part of its module design is borrowed from [MMDetection](https://github.com/open-mmlab/mmdetection), [DETR](https://github.com/facebookresearch/detr), and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR).








