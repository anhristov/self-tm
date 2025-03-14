# Self-TM [BDCC 2025]

<div align="center">
    <a href="https://www.mdpi.com/2504-2289/9/2/38"><strong>Paper</strong></a> |
    <a href="#usage"><strong>Usage</strong></a> |
    <a href="#checkpoints"><strong>Checkpoints</strong></a>
</div>

--------------------------------------------------------------------------------

Official code release for the paper: **Self-supervised foundation model for template matching**

<div align="center">
    <picture>
        <img src="Self-TM%20diagram.jpg" width="70%" height="70%" align="center">
    </picture>
</div>


**Abstract:** Finding a template location in query image is a fundamental problem in many computer vision applications, such as localization of known objects, image registration, image matching and object tracking. Currently available methods fail when not enough training data is available or big variations in the textures, different modalities and weak visual features exist in the images, leading to limited applications on real-world tasks. We introduce Self-Supervised Foundation Model for Template Matching (Self-TM), a novel end-to-end approach for self-supervised learning template matching. The idea behind Self-TM is to learn hierarchical features incorporating localization properties from images without any annotations. As going deeper in the convolutional neural network (CNN) layers, their filters begin to react to more complex structures and their receptive fields increase. This leads to loss of localization information in contrast to the early layers. The hierarchical propagation of the last layers activations back to the first layer results in precise template localization. Due to its zero-shot generalization capabilities on tasks such as image retrieval, dense template matching and sparse image matching our pre-trained model can be classified as a foundation one.

# Usage
inference.py contains example usage of the SelfTM matching. To try with your own images, change:
```sh
24   query_image_filename = './data/input_image.png'
25   template_image_filename = './data/template.png' 
```
with your own filepaths.

# Checkpoints
 - [SelfTM-Base (trained on ImageNet)](https://u.pcloud.link/publink/show?code=XZ8WJ85ZoSzl6mh69kfzB3MuYJaipyt8kWEX)
 - [SelfTM-Base (trained on ImageNet, finetunned on HPatches)](https://u.pcloud.link/publink/show?code=XZRsJ85ZFESA1GChrc0zjI6veBAjxLHH6zRX) 

# Citation
If you find this repository useful, please cite:
```
MDPI and ACS Style
Hristov, A.; Dimov, D.; Nisheva-Pavlova, M. Self-Supervised Foundation Model for Template Matching. Big Data Cogn. Comput. 2025, 9, 38. https://doi.org/10.3390/bdcc9020038

AMA Style
Hristov A, Dimov D, Nisheva-Pavlova M. Self-Supervised Foundation Model for Template Matching. Big Data and Cognitive Computing. 2025; 9(2):38. https://doi.org/10.3390/bdcc9020038

Chicago/Turabian Style
Hristov, Anton, Dimo Dimov, and Maria Nisheva-Pavlova. 2025. "Self-Supervised Foundation Model for Template Matching" Big Data and Cognitive Computing 9, no. 2: 38. https://doi.org/10.3390/bdcc9020038

APA Style
Hristov, A., Dimov, D., & Nisheva-Pavlova, M. (2025). Self-Supervised Foundation Model for Template Matching. Big Data and Cognitive Computing, 9(2), 38. https://doi.org/10.3390/bdcc9020038
```
