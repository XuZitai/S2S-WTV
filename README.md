## Demo code for:


[[Paper]](https://ieeexplore.ieee.org/document/10105620)
[[Demo code]](https://github.com/XuZitai/S2S-WTV/blob/main/S2S_WTV.py)
# Unsupervised 3D Random Noise Attenuation Using Deep Skip Autoencoder  


### Citation

If you use this model in your research, please cite:

    @ARTICLE{xu2023,
    author={Xu, Zitai and Luo, Yisi and Wu, Bangyu and Meng, Deyu},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={S2S-WTV: Seismic Data Noise Attenuation Using Weighted Total Variation Regularized Self-Supervised Learning}, 
    year={2023},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TGRS.2023.3268554}}
     

### Abstract

Seismic data often undergoes severe noise due to environmental factors, which seriously affect subsequent applications. Traditional hand-crafted denoisers such as filters and regularizations utilize interpretable domain knowledge to design generalizable denoising techniques, while their representation capacities may be inferior to deep learning denoisers, which can learn complex and representative denoising mappings from abundant training pairs. However, due to the scarcity of high-quality training pairs, deep learning denoisers may sustain some generalization issues over various scenarios. In this work, we propose a self-supervised method that combines the capacities of the deep denoiser and the generalization abilities of the hand-crafted regularization for seismic data noise attenuation. Specifically, we leverage the Self2Self (S2S) learning framework with a trace-wise masking strategy for seismic data denoising by solely using the observed noisy data. Parallelly, we suggest the weighted total variation (WTV) to further capture the horizontal local smooth structure of seismic data. Our method, dubbed as S2S-WTV, enjoys both high representation abilities brought from the self-supervised deep network and good generalization abilities of the hand-crafted WTV regularizer and the self-supervised nature. Therefore, our method can effectively remove noise and preserve the fine details of the seismic signal. To tackle the S2S-WTV optimization model, we introduce an alternating direction multiplier method (ADMM)-based algorithm. Extensive experiments on synthetic and field noisy seismic data demonstrate the effectiveness of our method as compared with state-of-the-art traditional and deep learning-based seismic data denoising methods.
 

**Note**

Pytorch
