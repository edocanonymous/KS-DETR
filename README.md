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
<td align="center"> <a href="">model</a></td>
</tr>

<!-- ROW: ks_dab_detr_r101_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_dab_detr/ks_dab_detr_r101_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-DAB-DETR-R101</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">45.3</td>
<td align="center"> <a href="">model</a></td>

<!-- ROW: ks_dab_detr_swin_t_in1k_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_dab_detr/ks_dab_detr_swin_tiny_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-DAB-DETR-Swin-T</a></td>
<td align="center">Swin-T</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">47.1</td>
<td align="center"> <a href="">model</a></td>
</tr>




<!-- ROW: ks_conditional_detr_r50 -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_conditional_detr/ks_conditional_detr_r50_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-Conditional-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">45.3</td>
<td align="center"> <a href="">model</a></td>

<!-- ROW: ks_conditional_detr_r101_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_conditional_detr/ks_conditional_detr_r101_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-Conditional-DETR-R101</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">47.1</td>
<td align="center"> <a href="">model</a></td>
</tr>


<!-- ROW: ks_dn_detr_r50_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_dn_detr/ks_dn_detr_r50_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-DN-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">45.2</td>
<td align="center"> <a href="">model</a></td>

<!-- ROW: ks_dn_detr_r101_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_dn_detr/ks_dn_detr_r101_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py">
KS-DN-DETR-R101</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">46.5</td>
<td align="center"> <a href="">model</a></td>
</tr>



<!-- ROW:  -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_deformable_detr/ks_deformable_detr_r50_12ep_smlp_qkv_triple_attn_outproj_ffn_v0.py">
KS-Deformable-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">36.4</td>
<td align="center"> <a href="">model</a></td>

<!-- ROW:  -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_deformable_detr/ks_deformable_detr_r101_12ep_smlp_qkv_triple_attn_outproj_ffn_v0.py">
KS-Deformable-DETR-R101</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">38.4</td>
<td align="center"> <a href="">model</a></td>
</tr>

 <tr><td align="left"><a href="projects/ks_detr/configs/ks_dn_deformable_detr/ks_dn_deformable_detr_r50_12ep_smlp_qkv_triple_attn_outproj_ffn_v0.py">
KS-DN-Deformable-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">12</td>
<td align="center">46.5</td>
<td align="center"> <a href="">model</a></td>

<!-- ROW: ks_deformable_detr_r50_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_deformable_detr/ks_deformable_detr_r50_50ep_smlp_qkv_triple_attn_outproj_ffn_v0.py">
KS-Deformable-DETR-R50</a></td>
<td align="center">R-50</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">44.8</td>
<td align="center"> <a href="">model</a></td>


<!-- ROW: ks_deformable_detr_r101_50ep -->
 <tr><td align="left"><a href="projects/ks_detr/configs/ks_deformable_detr/ks_deformable_detr_r101_50ep_smlp_qkv_triple_attn_outproj_ffn_v0.py">
KS-Deformable-DETR-R101</a></td>
<td align="center">R-101</td>
<td align="center">IN1k</td>
<td align="center">50</td>
<td align="center">46.0</td>
<td align="center"> <a href="">model</a></td>
</tr>


</tbody></table>



## What's New



## Installation

```shell

conda create -n ksdetr python=3.8 -y
conda activate ksdetr

git clone https://github.com/edocanonymous/KS-DETR.git
cd KS-DETR
python -m pip install -e detectron2
pip install -e .

```

## Getting Started



## Training
All configs can be trained with:
```bash
cd detrex
python tools/train_net.py --config-file projects/dab_detr/configs/path/to/config.py --num-gpus 8
```



To train `KS-DAB-DETR-R50`, `KS-DAB-DETR-R101`, and `KS-DAB-DETR-Swin-T`,
```bash
python tools/train_net.py --config-file projects/ks_detr/configs/ks_dab_detr/ks_dab_detr_r50_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py --num-gpus 8

python tools/train_net.py --config-file projects/ks_detr/configs/ks_dab_detr/ks_dab_detr_r101_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py --num-gpus 8

python tools/train_net.py --config-file projects/ks_detr/configs/ks_dab_detr/ks_dab_detr_swin_tiny_50ep_smlp_qkv_triple_attn_share_outproj_ffn.py --num-gpus 8

```


## Evaluation
Model evaluation can be done as follows:
```bash
cd detrex
python tools/train_net.py --config-file projects/dab_detr/configs/path/to/config.py --eval-only train.init_checkpoint=/path/to/model_checkpoint
```

For instance, 


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Acknowledgement
- Our code is built on detrex, which is an open-source toolbox for Transformer-based detection algorithms created by researchers of **IDEACVR**. 

- detrex is built based on [Detectron2](https://github.com/facebookresearch/detectron2) and part of its module design is borrowed from [MMDetection](https://github.com/open-mmlab/mmdetection), [DETR](https://github.com/facebookresearch/detr), and [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR).








