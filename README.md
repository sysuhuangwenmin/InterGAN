# InterGAN
An PyTorch implementation of Interactive Generative Adversarial Networks with High-Frequency Compensation for Facial Attribute Editing

##Abstract
Recently, facial attribute editing has drawn increasing attention and achieve significant progress due to Generative Adversarial Network (GAN).  Since paired images  before and after editing are not available, existing methods typically perform the editing and reconstruction tasks simultaneously, and  transfer  facial details learned from the reconstruction  to the editing  via sharing latent representation space and weights. In addition, they usually  introduce  skip connections between the encoder and decoder to improve  image quality  at the cost of attribute editing ability. Unlike existing methods, we propose a novel method called Interactive GAN  (InterGAN) with high-frequency compensation for facial attribute editing in this paper. Specifically,  we first propose the cross-task interaction (CTI) to  fully explore the relationships between editing and reconstruction tasks. The CTI includes  two translations:  style translation adjusts the mean and variance of feature maps according to style features, and conditional translation utilizes attribute vector as condition to guide feature map transformation.  They  provide effective information interaction for improving editing and reconstruction performance.
Without using skip connections between the encoder and decoder,  furthermore,  we employ the  high-frequency compensation module (HFCM) in the encoder.  The HFCM tries to collet potentially loss information from input images and each down-sampling layers of the encoder, and then re-inject them into subsequent layers of the encoder to alleviate the information loss.  Extensive qualitative and quantitative experiments evaluated on CelebA-HQ  demonstrate that the proposed  method  outperforms state-of-the-art methods both in  attribute editing accuracy and image quality.  In addition,  ablation experiments show the effectiveness of CTI and HFCM in the proposed model.

## Introduction
### Some facial examples of single and multiple attribute editing generated by the proposed InterGAN. Zoom in for better resolution.
![result](https://raw.githubusercontent.com/sysuhuangwenmin/InterGAN/main/images/result.png)


## Model
### The network architecture of the proposed InterGAN.
![InterGAN](https://raw.githubusercontent.com/sysuhuangwenmin/InterGAN/main/images/InterGAN.png)

### Structure of the CTI module.
![CTI](https://raw.githubusercontent.com/sysuhuangwenmin/InterGAN/main/images/CTI.png)


### Comparison on the proposed InterGAN with and without HFCM. Zoom in for better resolution.
![HFCM](https://raw.githubusercontent.com/sysuhuangwenmin/InterGAN/main/images/HFCM.png)


## Main results
### An example of editing Blond Hair with different models. Zoom in for better resolution.
![single_zoom](https://raw.githubusercontent.com/sysuhuangwenmin/InterGAN/main/images/single_zoom.png)

### Results of single facial attribute editing. Zoom in for better resolution.
![single](https://raw.githubusercontent.com/sysuhuangwenmin/InterGAN/main/images/single.png)

### Results of multiple facial attribute editing. Zoom in for better resolution.
![multi](https://raw.githubusercontent.com/sysuhuangwenmin/InterGAN/main/images/multi.png)

### Interpolation results. Zoom in for better resolution.
![interpolation](https://raw.githubusercontent.com/sysuhuangwenmin/InterGAN/main/images/interpolation.png)

### Visual comparison of edited images generated by ``InterGAN w/o HFCM'' and InterGAN. Zoom in for better resolution.
![HFCM2](https://raw.githubusercontent.com/sysuhuangwenmin/InterGAN/main/images/HFCM2.png)

### Visual comparison of edited images generated by ``InterGAN w/o CTI'' and InterGAN. Zoom in for better resolution.
![CTI2](https://raw.githubusercontent.com/sysuhuangwenmin/InterGAN/main/images/CTI2.png)

## Acknowledgement
This code refers to the following two projects:

[1] https://github.com/elvisyjlin/AttGAN-PyTorch

[2] https://github.com/rucmlcv/L2M-GAN

