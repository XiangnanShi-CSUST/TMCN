# TMCN
Codes for ***TMCN: Text-guided Mamba-CNN dual-encoder network for infrared and visible image fusion***

Jianming Zhang, Xiangnan Shi, Zhijian Feng, Yan Gui, Jin Wang.

-[*[Paper]*](https://www.sciencedirect.com/science/article/pii/S1350449525001884?via%3Dihub)  




## Citation
Export citation to BibTeX
```
@article{ZHANG2025105895,
title = {TMCN: Text-guided Mamba-CNN dual-encoder network for infrared and visible image fusion},
journal = {Infrared Physics & Technology},
volume = {149},
pages = {105895},
year = {2025},
issn = {1350-4495},
doi = {https://doi.org/10.1016/j.infrared.2025.105895},
url = {https://www.sciencedirect.com/science/article/pii/S1350449525001884},
author = {Jianming Zhang and Xiangnan Shi and Zhijian Feng and Yan Gui and Jin Wang},
keywords = {Image fusion, Text-guided, Mamba, Invertible neural networks},
abstract = {Infrared and visible image fusion (IVF) combines the complementary advantages of two images from different physical imaging methods to create a new image with richer information. To better address issues such as weak texture details, low contrast, and poor visual perception of overexposed and underexposed areas, we propose a text-guided Mamba-CNN dual-encoder network (TMCN). Firstly, to leverage the feature extraction capabilities of Mamba and CNN, we design a pre-training network to train a Mamba-based encoder, a CNN-based encoder, and a decoder. The structures of these encoders are also used in the image fusion stage. Then, we introduce a hybrid Mamba-CNN dual-encoder to extract global and local features from infrared and visible images, resulting in four distinct types of feature information. Secondly, we design a global fusion block (GFB) via the Mamba-based encoder, and a local fusion block (LFB) via the CNN-based encoder, to fuse the global and local features of the two modalities, respectively. Following these fusion blocks, we introduce text semantic information and utilize its stable and targeted characteristics to better solve the above problems. Therefore, we propose a plug-and-play text-guided block (TB) that first uses a CLIP-based text encoder to encode the input text, and then exploits feed-forward neural network (FFN) to extract two parameters for subsequent linear transformations, which reflect the text-guided mechanism. Finally, numerous experiments demonstrate that our method achieves excellent performance in IVF and has strong versatility. Furthermore, our method enhances the performance of downstream tasks such as object detection and semantic segmentation. The code will be available at https://github.com/XiangnanShi-CSUST/TMCN.}
}
```
Export citation to text

Jianming Zhang, Xiangnan Shi, Zhijian Feng, Yan Gui, Jin Wang. TMCN: Text-guided Mamba-CNN dual-encoder network for infrared and visible image fusion. Infrared Physics and Technology, 2025, vol. 149, 105895. DOI: 10.1016/j.infrared.2025.105895.  

## Abstract

Infrared and visible image fusion (IVF) combines the complementary advantages of two images from different physical imaging methods to create a new image with richer information. To better address issues such as weak texture details, low contrast, and poor visual perception of overexposed and underexposed areas, we propose a text-guided Mamba-CNN dual-encoder network (TMCN). Firstly, to leverage the feature extraction capabilities of Mamba and CNN, we design a pre-training network to train a Mamba-based encoder, a CNN-based encoder, and a decoder. The structures of these encoders are also used in the image fusion stage. Then, we introduce a hybrid Mamba-CNN dual-encoder to extract global and local features from infrared and visible images, resulting in four distinct types of feature information. Secondly, we design a global fusion block (GFB) via the Mamba-based encoder, and a local fusion block (LFB) via the CNN-based encoder, to fuse the global and local features of the two modalities, respectively. Following these fusion blocks, we introduce text semantic information and utilize its stable and targeted characteristics to better solve the above problems. Therefore, we propose a plug-and-play text-guided block (TB) that first uses a CLIP-based text encoder to encode the input text, and then exploits feed-forward neural network (FFN) to extract two parameters for subsequent linear transformations, which reflect the text-guided mechanism. Finally, numerous experiments demonstrate that our method achieves excellent performance in IVF and has strong versatility. Furthermore, our method enhances the performance of downstream tasks such as object detection and semantic segmentation. 
## 🌐 Usage

### ⚙ Network Architecture

Our TMCN is implemented in ``net.py``.

### ⚙ Test

Run 
```
python test.py
```
