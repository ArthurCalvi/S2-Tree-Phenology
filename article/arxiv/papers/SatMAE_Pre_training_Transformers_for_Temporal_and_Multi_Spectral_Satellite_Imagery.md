## **SatMAE: Pre-training Transformers for Temporal** **and Multi-Spectral Satellite Imagery**

**Yezhen Cong** _[∗]_ **Samar Khanna** _[∗]_ **Chenlin Meng** **Patrick Liu**
```
 yzcong@stanford.edu samar.khanna@stanford.edu

```

**Erik Rozi** **Yutong He** **Marshall Burke** **David B. Lobell** **Stefano Ermon**


Stanford University


**Abstract**


Unsupervised pre-training methods for large vision models have shown to enhance
performance on downstream supervised tasks. Developing similar techniques for
satellite imagery presents significant opportunities as unlabelled data is plentiful
and the inherent temporal and multi-spectral structure provides avenues to further
improve existing pre-training strategies. In this paper, we present SatMAE, a
pre-training framework for temporal or multi-spectral satellite imagery based on
Masked Autoencoder (MAE). To leverage temporal information, we include a
temporal embedding along with independently masking image patches across
time. In addition, we demonstrate that encoding multi-spectral data as groups of
bands with distinct spectral positional encodings is beneficial. Our approach yields
strong improvements over previous state-of-the-art techniques, both in terms of
supervised learning performance on benchmark datasets (up to _↑_ 7%), and transfer
learning performance on downstream remote sensing tasks, including land cover
classification (up to _↑_ 14%) and semantic segmentation. Code and data are available
on the project website: `[https://sustainlab-group.github.io/SatMAE/](https://sustainlab-group.github.io/SatMAE/)`


**1** **Introduction**


In recent years, self-supervised learning techniques have quickly become the norm for pre-training
models on large-scale natural image datasets [ 1, 2, 3, 4, 5, 6, 7, 8 ], and have demonstrated strong
performance on downstream tasks including image classification [ 3, 4, 9, 10 ], image segmentation [ 3,
11 ], representation learning [ 12, 13, 14 ], image compression [ 12, 15 ], image reconstruction [ 1 ], and
image generation [ 16 ]. Unlike supervised learning approaches, self-supervised learning techniques do
not require human labeling, making them appealing in settings where unlabeled data are abundant but
labeled data are scarce, such as remote sensing data (e.g., satellite imagery). While several large-scale
satellite image datasets have been carefully curated in the past few years, including Functional Map
of the World (fMoW) [ 17 ], BigEarthNet [ 18 ], xView [ 19 ], SpaceNet [ 20 ], annotating these datasets
requires specialized skills and is more expensive than traditional computer vision datasets. Moreover,
automatic analysis of satellite imagery is often needed for tasks with large societal impact such
as poverty or crop yield prediction [ 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 ], where acquiring large
amounts of labeled data through surveys is impossible or prohibitively expensive. This suggests that
self-supervised learning approaches for satellite imagery could be especially valuable.


However, existing self-supervised learning approaches [ 1, 2, 3, 4, 5, 6 ] are mainly designed for natural
images. As opposed to natural images such as ImageNet [ 31 ], satellite imagery is usually associated


_∗_ Equal contribution. Order determined via coin flip.


36th Conference on Neural Information Processing Systems (NeurIPS 2022).


Positional encoding across
spatial _and_ temporal/spectral


dimension


encoder decoder



input



reconstruction



patch embed + mask + pos enc. decoder pos enc.


Figure 1: With carefully-designed masking strategies across mutli-spectral and temporal images, and temporal
and spectral positional encodings, our SatMAE serves as a powerful SSL vision learner for remote sensing tasks.


with meaningful geographical and temporal information, and can consist of multiple spectral bands
representing sensor readings besides visible light (i.e., RGB channels typical in natural images).
Depending on the data source, satellite imagery can also vary significantly in resolution [ 32, 33 ].
While self-supervised learning methods for satellite imagery exist [ 34, 35 ], these approaches cannot
learn general representations for both temporal and multi-spectral remote sensing data.


To address this issue we propose **SatMAE**, a self-supervised learning framework based on masked
autoencoders (MAEs) [ 1 ] which naturally handles temporal and multi-spectral input data. We show
that introducing a positional encoding for the temporal/spectral dimension and _independently_ masking
patches across the temporal/spectral dimension benefits pre-training, allowing the model to learn
representations of the data that are more conducive to finetuning. Specifically, our contributions are:


1. We propose a novel method to leverage temporal or multi-spectral information in satellite
imagery to improve self-supervised pre-training with masked autoencoders (see 4).

2. We introduce fMoW-Sentinel, a new Sentinel-2 dataset cross-referenced with fMoW, as a
benchmark for training models on multi-spectral satellite imagery (see 5.1).

3. We demonstrate the effectiveness of pre-training transformers [ 36 ] on satellite imagery,
achieving significant improvement over previous state-of-the-art methods on benchmark
datasets as well as downstream remote sensing tasks (see 5)


**2** **Related Work**


**ML for SITS** Deep learning has been used for many Satellite Image Time Series (SITS) supervisedlearning tasks such as crop-type mapping [ 29, 28, 37, 38 ], yield prediction [ 39, 40 ], understanding the
economy [ 41, 42, 43, 44 ], precipitation forcasting [ 45 ], and land-cover classification [ 46, 47, 48, 27 ].
These works establish the usefulness of tailoring architectures such as LSTMs, self-attention, and
transformers to temporal data. However, outside of their specific task, they are often not directly
applicable to other remote-sensing datasets.


**SSL for Satellite Imagery** Self-supervised learning [ 2, 3, 4, 5, 6 ] has emerged as a promising
approach in remote sensing domains. For instance, [ 34 ] and [ 35 ] propose incorporating spatially
aligned images over time for contrastive self-supervised learning. Despite promising results, these
two contrastive learning approaches rely heavily on the quality of positive pairs, which is often
hard to control. [49] combines different sensor channels to generate co-located images that serve as
positive pairs. [ 50, 51, 52 ] apply off-the-shelf contrastive learning algorithms to satellite images. [ 52 ]
utilizes image inpainting and transformation prediction as additional pretext tasks. [ 53 ] leverages
geographical knowledge to aid SSL, which, however, can be difficult to obtain as annotations.


2


**Masked Autoencoder** MAE [ 1 ] is a recent powerful self-supervised learning method. Instead of
constructing a contrastive objective, it proposes the pretext task of reconstructing masked patches
of the input, and largely avoids the need for designing specific data augmentation. Inspired by
MAE’s state-of-the-art performance on a wide collection of vision benchmarks [ 1 ], many follow-up
works extend MAE to different data modalities. VideoMAE [ 54 ] proposes video tube masking
and reconstruction as a pretext task for video analysis. GMAE [ 55 ] adapts MAE to the domain of
graphs. MultiMAE [ 56 ] takes optional inputs of different modalities and accordingly includes other
training objectives to facilitate multi-modality learning. However, these works fail to optimally handle
temporal and multi-spectral input. VideoMAE requires equally-spaced image frames in the temporal
dimension, which is not the case for satellite data given the temporal irregularity and discontinuity in
sampling images of a location. In this work, we incorporate temporal and spectral information into a
masked autoencoder architecture, and propose a novel self-supervised framework for satellite data.


**3** **Background**


**Masked Autoencoder** The MAE is an autoencoder with asymmetrical encoding and decoding
stages [ 1 ]. It operates on images _I ∈_ R _[C][×][H][×][W]_, where _H, W_ are the height and width of the
image, respectively, and _C_ is the number of channels. The input image _I_ is resized to a sequence
of non-overlapping patches, _S ∈_ R _[L][×][P]_ [ 2] _[C]_, where _P_ is the height and width of the patch, and
_L_ = ( _H/P_ ) _·_ ( _W/P_ ) is the number of patches. Each patch is passed through a patch embedding
_f_ _p_ : R _[P]_ [ 2] _[C]_ _�→_ R _[D]_ to create a sequence _S_ _[′]_ _∈_ R _[L][×][D]_ of embedded patch “tokens”. A fraction _p_ _m_ of
the _L_ tokens are masked and only the remaining (1 _−_ _p_ _m_ ) _L_ “visible" patch tokens are fed to the
encoder, a Vision Transformer (ViT) [ 36 ] with positional embeddings to capture the spatial location
of the patch in the image. The decoder is a series of transformer blocks that operates on all _L_ tokens
(with positional embeddings added), where the _p_ _m_ _L_ encoded visible patches are placed in their
original sequence position among (1 _−_ _p_ _m_ ) _L_ masked patches represented by a learnable mask token.
The decoder outputs a reconstructed image _I_ [ˆ] _∈_ R _[C][×][H][×][W]_, which is compared to the original image
using the mean-squared error (MSE) loss, computed per-pixel only on the masked patches [1].


**Positional encoding** Positional encoding allows transformers to make their learned representations
position-aware. In MAE [1] and in many transformers [57, 58], the positional encoding is:



_k_
`Encode` ( _k,_ 2 _i_ ) = sin



_k_
_di_ _[,]_ `[ Encode]` [(] _[k,]_ [ 2] _[i]_ [ + 1) = cos] Ω



2 _i_
Ω _d_



2 _i_
Ω _d_



_i_ (1)

_d_



Here, _k_ is the position, _i_ is the index of feature dimension in the encoding, _d_ is the number of possible
positions, and Ω is a large constant (normally set to 10000). In MAE, position is defined as the index
of the patch along the x or y axes. Therefore, _k_ ranges from 0 to _H/P_ (or _W/P_ ). The final encoding
is generated by concatenating the encodings of the x and y coordinates.


**4** **Method**


In this section, we describe SatMAE with temporal (4.1) and multi-spectral (4.2) satellite images.


**4.1** **Temporal SatMAE**


We now consider input tensors _I_ _T_ _∈_ R _[T][ ×][C][×][H][×][W]_, where _T_ denotes the number of images in a
temporal sequence. In video data, _T_ frames are usually equally spaced. However, temporal satellite
imagery rarely has images at regular intervals. More commonly, several snapshots, or versions, of a
given location are taken at irregular times. The length and sample frequency of these sequences of
satellite images vary drastically over years and across different regions.


Naïvely, one could reshape _I_ _T_ to _I_ _T_ _[′]_ _[∈]_ [R] _[T C][×][H][×][W]_ [, effectively concatenating the temporal sequence]
of images along the spectral (i.e. channel) dimension, and then apply the MAE machinery verbatim.
This method poses a few difficulties: (i) the model may be unable to generalise to a temporal ordering
different to the one used in pre-training, since it can only understand order through the position of
images in the stacked-timeseries (ii) the model cannot reason about the length of time separating
two consecutive images in a time sequence, which may be variable when images of a location are


3


sampled at irregular intervals (iii) the model loses access to temporal fine-grained information in
deeper layers, as its only direct exposure to encode temporal information is through the initial patch
embedding _f_ _p_ (iv) the model is not temporally-shift invariant (i.e. the model would need to separately
learn to detect the same event in two different segments of a temporal sequence).


To address these challenges and to avoid losing temporal information, we resize the temporal sequence
_I_ _T_ to _S_ _T_ _∈_ R _[L]_ _[T]_ _[ ×][P]_ _[T]_ _[ P]_ [ 2] _[C]_, where _L_ _T_ = _L·_ ( _T/P_ _T_ ) = ( _H/P_ ) _·_ ( _W/P_ ) _·_ ( _T/P_ _T_ ), _P_ _T_ is the “patch size”
in the temporal dimension, and _L_ and _P_ are defined in 3. Prior works using transformers for video
data suggest using _P_ _T_ = 2, where each “patch” is a cube of shape 2 _×_ 16 _×_ 16 [ 54, 59, 60 ]. Since our
data has much shorter temporal sequence lengths [ 17 ], we let _P_ _T_ = 1 such that _L_ _T_ = _L · T_ . In order
to operate on inputs of any temporal order, we re-use the same patch embedding _f_ _p_ : R _[P]_ [ 2] _[C]_ _�→_ R _[D]_

for each image in the time series, giving us an embedded sequence of tokens _S_ _T_ _[′]_ _[∈]_ [R] _[L]_ _[T]_ _[ ×][D]_ [.]


**4.1.1** **Temporal Encoding**


For each embedded token in the _L_ _T_ length sequence, we
need to ensure the model retains information about its
spatial and temporal position. As shown in many prior
works [ 34, 35 ], the timestamp of a satellite image is useful
for many pre-training or downstream vision tasks. We
propose a temporal encoding scheme compatible with the
masked autoencoder architecture by treating the temporal
dimension similarly to the positional dimensions (see 3).



Figure 2: Top: Encoding each temporal patch
with a shared patch embedding _f_ _p_ . Bottom:
Encoding each spectral patch with a different
patch embedding _f_ _p_ _j_ for each group _j_ .



The timestamp of a satellite image is represented as “yearmonth-day-hour-minute-second”. Instead of passing the
entire numerized timestamp into a feature encoder, we
propose only keeping the useful parts. Intuitively, the day,
minute, and second should be unrelated to the visual appearance of a region. Thus, including these components in
the temporal encoding may not be beneficial, and can even
be detrimental. In contrast, a landscape may evolve over
years due to weather, geology, and human activity. The
month reflects season and climate, and the hour reflects
daylight and temperature.


Then, the temporal encoding is formulated as:



_t_ _k,i_ = CONCAT[ `Encode` ( _k_ year _, i_ ) _,_ `Encode` ( _k_ month _, i_ ) _,_ `Encode` ( _k_ hour _, i_ )] (2)


And the final encoding is generated by concatenating the temporal encoding to the positional encoding
defined in 3 such that the total length of the encoding is _D_ .


**4.1.2** **Masking Strategies**


With an additional temporal dimension, masking a subset of the _L_ _T_ tokens needs to be treated with
care. As seen in figure 3, there are different ways to mask a temporal series of satellite images.


**Consistent Masking** Each image is “patchified” separately, but the masked regions are consistent
across all images (fig. 3a). This approach is also used in VideoMAE [54], with video input.


**Independent Masking** Each image is “patchified” separately, and masked regions may not be the
same across every image. Instead, a fraction _p_ _m_ of the full sequence of all patch tokens are masked.
Another variant is to independently mask the regions of each image, but keep the ratio _p_ _m_ of masked
regions fixed per image. Both variants are equivalent in expectation. Effectively, the model may look
at unmasked values of a region that is masked in one image but not in others. This setting may lead to
an easier task for video data since the model can “cheat” and exploit temporal redundancy in videos
with high framerates [ 54 ]. However, we argue that this form of “cheating” is less feasible in temporal
satellite imagery, given the strong impact of seasonal variation and changing human activity over
periods of time and the much larger time deltas between temporally consecutive images (see fig. 3a).


4


(a) Temporal data (b) Spectral data


Figure 3: 3a **Temporal masking** : For images in a timeseries, we can choose to keep a patch fully visible or fully
masked across time ( _consistent masking_ ), or independently mask all patches ( _independent masking_ ). In both
cases, a fraction _p_ _m_ patches are masked. Here, _T_ = 3, and the leftmost column orders the temporal sequence
according to the timestamp features. For example, “y-12, m-12, h-15” is 12 years from the minimum year (2002),
the zero-indexed month 2, and the 15th hour of the day; i.e., roughly 2014, March, 15:00. 3b **Spectral Masking** :
The same masking strategies are adapted to groups of the 13 spectral bands in Sentinel-2 images.


**Independent Masking + Inconsistent Cropping** During data pre-processing, we can crop square
regions for input inconsistently so that images in the same temporal sequence may be spatiallyunaligned. This strategy may help the model learn better representations as it may learn to align
images in the sequence across the spatial and temporal dimensions.


**4.2** **Multi-spectral SatMAE**


While MAE does operate on images _I ∈_ R _[C][×][H][×][W]_, usually _C_ = 3 for RGB images. Satellite data,
on the other hand, can often have multiple spectral bands. For example, Sentinel-2 imagery has
_C_ = 13 bands of 10m, 20m and 60m spatial resolution, each of different wavelengths (see A.2.2).
Below, we discuss and later experimentally compare various ways to encode spectral information.


**Stack Channels** The sequence of patches _S ∈_ R _[L][×][P]_ [ 2] _[C]_ is embedded to a sequence of tokens
_S_ _[′]_ _∈_ R _[L][×][D]_, thus treating the multi-band image as is. We denote this method **SatMAE+Stack** .


**Group Channels** There are limitations to naively stacking the spectral information, especially that
a single convolutional patch embedding may be insufficient to fully capture fine-grained information
present in multiple bands of different wavelengths and spatial resolution. We would like the model to
preserve information about the different bands through the encoding and decoding stages.


To address this limitation, we propose _grouping_ subsets of spectral bands. Given _C_ channels, we
form _G_ groups _g_ 1 _, g_ 2 _, . . ., g_ _G_ such that _g_ 1 + _g_ 2 + _· · ·_ + _g_ _G_ = _C_ . This is analogous to slicing the
image _I_ in the channel dimension, creating images _I_ 1 _, . . ., I_ _G_, where _I_ _j_ _∈_ R _[g]_ _[j]_ _[×][H][×][W]_ . We use a
separate patch embedding _f_ _p_ _j_ : R _[P]_ [ 2] _[g]_ _[j]_ _�→_ R _[D]_ for each group _j_, thus allowing the model to best
represent each possibly different group of channels as token embeddings. Therefore, each group _j_ is
first resized from _I_ _j_ _∈_ R _[g]_ _[j]_ _[×][H][×][W]_ to _S_ _j_ _∈_ R _[L][×][P]_ [ 2] _[g]_ _[j]_, and then each patch is embedded with _f_ _p_ _j_ to
produce a sequence of embedded tokens _S_ _j_ _[′]_ _[∈]_ [R] _[L][×][D]_ [. The sequences] _[ S]_ 1 _[′]_ _[, . . ., S]_ _G_ _[′]_ [are concatenated]
to produce the final set of tokens _S_ _[′]_ _∈_ R _[GL][×][D]_ .


**Spectral Encoding** Since the tokens in _S_ _[′]_ correspond to a patch location ( _m, n_ ) in the input image
and a group of channels _g_ _j_, we include an encoding for the group index _k_ _g_ similar to 4.1.1


_g_ _k_ _g_ _,i_ = `Encode` ( _k_ _g_ _, i_ ) (3)


Note that this encoding simply depends on a user-devised channel grouping, and differs from eq. (2)
since additional metadata for the imagery, like its date, is not needed. The final encoding is a


5


concatenation of the positional _x_ _k,i_, _y_ _k,i_ and the spectral encoding _g_ _k,i_ such that the total dimension
is _D_ (see fig. 2). This positional encoding is added to _S_ _[′]_ before inputting it to the encoder. We denote
the combined setting of grouping channels and using a group encoding as **SatMAE+Group** .


**Masking Strategies** We consider _consistent masking_ (denoted **SatMAE+Group+CM** ) and _inde-_
_pendent masking_ ( **SatMAE+Group+IM** ) as defined in section 4.1.2 and as visualized in fig. 3b.


**5** **Experiments**


In this section, we first introduce the datasets we considered, including a new multi-spectral remote
sensing image dataset for downstream task evaluation (5.1). We then present our results on benchmark
datasets (5.2, 5.3, 5.4) and various remote sensing transfer-learning and downstream tasks 5.5. For
all experiments, we compare with the current state-of-the-art methods [ 34, 35 ] and with supervised
learning from scratch using the ViT backbone of SatMAE. In summary, our approach demonstrates
strong performance on all the tasks we considered, yielding improvements over previous state-of-theart techniques by up to 6% on supervised learning benchmarks, and up to 14% on remote sensing
transfer-learning downstream remote sensing tasks.


**5.1** **Datasets for Pre-training**


**fMoW RGB** Functional Map of the World (fMoW) [ 17 ] is a dataset of high-resolution satellite
image time series across the world, with a task of classification among 62 categories.


**fMoW Sentinel** We create a new dataset based on the fMoW RGB dataset. We collect all 13
frequency bands provided by Sentinel-2 (B1-12 and B8A) for the original fMoW locations, at some
of the same times as fMoW images plus some extra times, for a total of 712,874 training images,
84,939 validation images, and 84,966 test images. More details are included in appendix A.1.


**5.2** **fMoW RGB (non-temporal)**



Method Backbone Frozen/Finetune

Sup.* ResNet50 -/69.05
Sup. _†_ ResNet50 -/69.07
GASSL [34] ResNet50 68.32/71.55
Sup.* ViT-Large -/62.48
Sup. _†_ ViT-Large -/75.70
Sup. _‡_ ViT-Large -/76.91
SatMAE ViT-Large 65.94/ **77.84**


Table 1: Top 1 Accuracy on fMoW classification.
**Frozen** : only performing linear classification on frozen
features of the pre-trained model. **Finetune** : end-to-end
finetuning the whole model. * is training from scratch,
and _†_ is using supervised-learning ImageNet weights,
and _‡_ is SSL MAE ImageNet weights.


**5.3** **fMoW RGB (temporal)**



In this section, we perform experiments on
fMoW single image classification task. Following [ 34 ], we report both the performance
of linear probing and finetuning setting. Table
1 shows that compared to the previous stateof-the-art self-supervised method using a contrastive momentum encoding approach [ 34, 3 ],
our SatMAE achieved a 6.29% improvement in
top 1 classification accuracy. Interestingly, without SatMAE pre-training the ViT-large model
could only reach 62.48% at convergence after
50 epochs of finetuning compared to 69.05%
achieved by training a ResNet-50 model from
scratch. This is likely because the ViT [ 36 ] backbone is harder to finetune from scratch than
ResNet50 [ 61 ], which makes the pre-trained
model more valuable.



**Main experiments** We perform image-sequence classification on the temporal version of fMoW
RGB to evaluate our temporal SatMAE. The temporal fMoW consists of co-located image sequences
with a length of 3. As seen in table 2, SatMAE surpasses the previous state-of-the-art by 4.48% and
improves the non-temporal result by 2.06% in top 1 classification accuracy. We also outperform
UTAE [ 48 ], a SITS state-of-the-art, by 18%. We can observe from rows 5-8 that this gain is not from
the larger model to handle sequences of data. Naively stacking the image sequences in the channel
dimension performs even worse than the non-temporal SatMAE. Again, SatMAE pre-training is
crucial for ViT to outperform ResNet50. Training details are in appendix A.3.2.


6


Method Backbone Top Acc. (1/5)
Sup.* ResNet50 73.24/SeCo [35] ResNet50 66.80/GASSL [34] ResNet50 74.11/UTAE [48] U-Net 61.59/86.45
Sup.* ViT-Large 61.89/84.23
SatMAE+Stack ViT-Large 75.85/88.68
MAE+Test Aug. ViT-Large 78.90/93.31
MAE _∥_ ViT-Large 76.78/92.01
SatMAE ViT-Large **81.49** / **93.26**


Table 2: Classification results on the temporal fMoW
RGB dataset. * means finetuning from scratch. _∥_ means
copying the input image 3 times instead of using temporal sequences as input. SatMAE+Stack here means
stacking the image sequence along the channel space.


Temp. Indep. Cons. Test
Enc. Mask. Crop. Aug. Top 1 Acc.

✓ ✓ 78.07

✓ ✓ 78.45

✓ ✓ 79.90

✓ ✓ ✓ 79.69

✓ ✓ ✓ ✓ **81.49**


Table 4: Ablation studies on different components of
temporal SatMAE on the temporal fMoW classification
task. The first column is whether using temporal encoding, the second is whether using independent masking,
the third is whether cropping consistently, and the last
one is whether applying test-time augmentation.



Method Backbone Top Acc. (1/5)
Sup. Learning* ResNet152 49.12/75.73
Sup. Learning _‡_ ResNet152 54.46/78.99
MoCo-v3 ViT-Base 50.45/76.37
MoCo-v3+Group ViT-Base 51.33/75.68
SatMAE+Group* ViT-Large 53.03/77.14
SatMAE+Group _†_ ViT-Large 51.61/77.26
SatMAE+Group _‡_ ViT-Large 47.57/72.26
SatMAE+Group _§_ ViT-Large 49.49/76.30
SatMAE+Stack ViT-Large 57.37/81.63
SatMAE+Group+IM ViT-Large 59.30/82.81
SatMAE+Group+IM ViT-Large **61.48** / **85.17**

Table 3: Top 1 & Top 5 Accuracy on the fMoW Sentinel validation set. The different initializations are: *
from scratch, _†_ MAE ImageNet weights, _‡_ supervised
ImageNet weights, _§_ SatMAE fMoW RGB weights.
Other rows use fMoW Sentinel for pre-training. The
last row includes additional data augmentations (5.4).


Group Indp. Spec.
Back. Strat. Mask. Enc. Top 1 Acc.


Base X ✓ ✓ 59.11
Large X ✓ 58.87
Large X ✓ 57.76
Large H ✓ ✓ 57.78
Large R ✓ ✓ 58.76
Large X ✓ ✓ **59.30**


Table 5: Ablation studies on spectral SatMAE on
fMoW-Sentinel. The first column denotes using ViTBase or ViT-Large. The second column is the grouping
strategy (see 5.4). The third column denotes independent or consistent masking. The last column is whether
the spectral group encoding 3 is used.



**Ablation studies** Table 4 provides a comprehensive ablation study on the components of temporal
SatMAE. We see that improved performance is mainly due to the temporal encoding and adopting
_independent masking_ rather than the consistent masking strategy suggested in VideoMAE [ 54 ].
Interestingly, consistent cropping slightly decreases performance, indicating that the model does
not rely on perfectly spatially-aligned image sequences. In addition, using test-time augmentations
similar to [ 34 ] is beneficial. Further ablations on mask ratio _p_ _m_ and patch size _P_ are in appendix A.4.


**5.4** **fMoW Sentinel (Multi-spectral)**


In this section, we pre-train and finetune SatMAE on the image classification task of the fMoWSentinel dataset. We pre-train SatMAE+Stack 4.2 and investigate SatMAE+Group+CM 4.1.2 and
SatMAE+Group+IM 4.1.2, (see 4.2, 4.2). The full models are then finetuned on the fMoW-Sentinel
image classification task. For comparison, we also finetune the ResNet-152 model [ 61 ] from scratch
and from a supervised ImageNet initialization. We pick the largest model, ResNet-152, for fairer
comparison with ViTs. We also include MoCo-v3 [ 62, 3 ], a popular SSL method. Given the
differences in applying RGB-image augmentations to satellite imagery, we implement two versions:
(i) MoCo-v3: we apply all of the same augmentations, except random grayscale and solarize, to
create 2 views of the 10-channel image. (ii) MoCo-v3+Group: we split the 10 bands into two groups
suggested by [ 2 ], and apply augmentations to each to create a positive pair of two 5-channel images.


**Model configuration** As not all of the 13 Sentinel-2 bands may be useful, in our experiments we
drop bands B1, B9 and B10, which correspond to a spatial resolution of 60m. Of the remaining
10 bands, we form three groups: (i) RGB+NIR: B2, B3, B4, B8 (ii) Red Edge: B5, B6, B7, B8A
(iii) SWIR: B11, B12. We choose this grouping to ensure each group has bands of the same spatial
resolution and similar wavelength (see A.2.2, A.6). Only the last row of table 3 includes additional
data augmentations used during finetuning as in [ 1 ]. See A.3.3 for pre-training and finetuning details.


7


Method Backbone Top 1 Acc.
Sup. (Scratch) ResNet50 54.46
GASSL [34] ResNet50 57.63
Sup. (Scratch) ViT-Large 69.65
SatMAE ViT-Large **71.77**


Table 6: NAIP land cover classification results.


Method Backbone Top 1 Acc.
Sup. (Scratch) ResNet18 63.21
Sup. (IN init.) ResNet18 86.44
GASSL [34] ResNet18 89.51
SeCo [35] ResNet18 93.14
SatMAE* ViT-Large 95.74
SatMAE ViT-Large 98.94
SatMAE+Group+IM ViT-Large **98.98**

Table 8: EuroSAT land cover classification results. *
means we only use the RGB channels of the data.



Method Backbone mIoU

Sup. (Scratch) ResNet50 75.57
GASSL [34] ResNet50 **78.51**
Sup. (Scratch) ViT-Large 74.71
SatMAE ViT-Large 78.07


Table 7: SpaceNet v1 building segmentation results.


Method Backbone mAP

Sup. (Scratch) ResNet50 69.49
Sup. (IN init.) ResNet50 80.04
GASSL [34] ResNet50 80.20
SeCo [35] ResNet50 **82.62**
Sup. (Scratch) ViT-Large 80.07
SatMAE ViT-Large 82.13


Table 9: BigEarthNet multi-label classification results.
Following [ 35 ], we use mean Average Precision (mAP)
as the metric, and use a newer set of class labels.



**Results** We present results in table 3. Our method SatMAE+Group+IM achieves the highest accuracy, outperforming supervised training from scratch ( _↑_ 6.27%) and ImageNet-initialized backbones
( _↑_ 4.84%). ImageNet initializations may be less useful than in fMoW-RGB given the larger distributional shift to multi-spectral input data. We also note the effectiveness of grouping channels over
processing all bands only at the patch embedding level (i.e. SatMAE+Stack).


**Ablation Studies** We investigate the design of SatMAE for multi-spectral data in table 5. For
grouping strategy, we implement alternate band groups to test the hypothesis that grouping bands
based on wavelength and resolution is beneficial. X represents the band groups in 5.4. H represents
splitting the 10 bands into two halves, {(2,3,4,5,6), (7,8,8A,11,12)}. R represents a random split into
three groups {(6,5,11,12), (8A,4,8,3), (7,2)}, reflecting the same group sizes as X. As seen, the choice
of band groups does influence performance, yielding a gain of about 0.6%. Moreover, ViT-Base
performs strongly, suggesting that SatMAE is the reason for improved performance rather than the
number of parameters in ViT. Interestingly, _independent masking_ performs the best, which prompts
the model to “peek” at unmasked band groups to reconstruct the same region in a masked band group.


We also include further experiments on the length of pre-training (see A.3.3), the impact of mask
ratio _p_ _m_ and patch size _P_ (see A.5), and the usefulness of the 13 Sentinel-2 spectral bands (see A.6).


**5.5** **Transfer Learning Experiments**


Now, we finetune our pre-trained SatMAE on downstream tasks on remote-sensing datasets, including
land cover classification (5.5), multi-label classification (5.5), and building segmentation (5.5).
Finetuning details are included in A.7, A.8, A.9, A.10.


**Land Cover Classification** We perform transfer learning experiments on land cover classification
using the NAIP and EuroSAT [ 63 ] dataset. NAIP consists of RGB+CIR images of 66 land cover
classes obtained by the USDA’s National Agricultural Imagery Program, which are split into 244,471
training and 55,529 validation images. EuroSAT is a small dataset containing 27,000 13-band satellite
images of 10 classes based on Sentinel-2. We follow [35, 64] for the train/val splits on EuroSAT.


Table 6 and table 8 shows the remarkable improvement of our SatMAE over the state-of-the-arts.
Although using the ViT-Large backbone already achieved good results, initializing the model with
SAT-MAE pre-trained weights further increased the accuracy by 2%-3%.


**Multi-label Classification** We also use the BigEarthNet [ 18 ] dataset for multi-label classification,
which consists of 13-band Sentinel-2 images of 19 classes in total. There are 354,196 images for
training and 118,065 images for validation. Following [35], we use a 10% subset of the train set.


8


Table 9 shows SatMAE pre-training improves upon the model trained from scratch by over 2%, and
achieves comparable results to the state-of-the-art. GASSL and SeCo were actually trained on a larger
pre-train dataset (1M Sentinel-2 images v.s. 713k) and with all 13 bands than our fMoW Sentinel.
Therefore we expect further improvement when we pre-train SatMAE with more data and for longer.


**Building Segmentation** In this section, we evaluate SatMAE on the semantic segmentation downstream task of the SpaceNet v1 dataset [ 20 ]. The SpaceNet v1 dataset consists of 6940 high resolution
satellite images with segmentation masks for buildings, which are divided into train and test sets of
5000 and 1940 images, respectively.


The results in table 7 show that our method achieves a larger performance gain from supervised
learning from scratch compared to [ 34 ]. The incompatibility of the ViT backbone with PSANet could
explain why the baseline performance is not as strong as that of using a ResNet50 backbone.


**5.6** **Visualizing reconstruction quality for SatMAE**


Figure 4: Reconstruction quality of SatMAE+IM (left) vs. SatMAE+CM (right). Further results in appendix C.


We show the visualization of the reconstruction quality of two different SatMAE masking strategies in
fig. 4. SatMAE+IM successfully reconstructs all the airplanes even though their number varies across
time. In contrast, the SatMAE with Consistent Masking missed some airplanes in the reconstruction.


**6** **Conclusion**


In this paper, we propose a new SSL framework based on the MAE architecture [ 1 ] tailored to remotesensing data (satellite imagery). Our novel masking strategy in a joint positional, temporal/spectral
space, along with the temporal and spectral encoding, enables our model to handle temporal and
multi-spectral satellite images as input and learn useful representations. Experiments on the datasets
for pre-training and multiple downstream datasets demonstrate the effectiveness of our pre-trained
SatMAE model, outperforming previous state-of-the-art results by large margins.


In the future, it would be useful to design more efficient transformer architectures. While SatMAE
has a similar number of parameters for both the temporal and multi-spectral setting as a regular ViT,
the increased length of token sequences can strain computational resources. Moreover, it is also worth
exploring optimal positional encodings for spectral and temporal data, as well as optimal groups of
spectral bands, either by neural-based search methods, or using prior knowledge. Lastly, investigating
better architectures for object detection and semantic segmentation using ViTs will be important in
generalising SatMAE to further downstream tasks.


9


**Broader Impact**


Accurate measurements of economic, social, and environmental phenomena are key inputs into policy
decisions made around the world, but the sparsity of labelled data on many outcomes means that
such decisions are often not guided by timely or accurate data. We demonstrate how a pre-training
framework could relieve the dependence on labelled data for many downstream tasks that use satellite
imagery as input. We hope our SatMAE method will help close the gap between SSL performance
on natural imagery and on the more challenging satellite imagery, and prompt further attention from
the ML community on the usefulness of SSL in satellite-imagery-related tasks.


Better extraction of information from satellite imagery has profound implications for our ability to
measure and understand a broad array of social, economic and environmental phenomena that are
critical for decision making. Our approach further amplifies the usefulness of the sparse amount of
labelled data that exist on key human outcomes, and could enable rapid and accurate extraction of
imagery features relevant for critical downstream tasks, including poverty prediction, infrastructure
development, and population estimation. Such information could aid governments in more rapid and
data-informed decision making and ultimately bring large societal benefits.


**7** **Acknowledgements**


This research is based upon work supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via 2021-2011000004,
HAI, NSF(#1651565), AFOSR (FA95501910024), ARO (W911NF-21-1-0125) and Sloan Fellowship.
The views and conclusions contained herein are those of the authors and should not be interpreted
as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or
the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for
governmental purposes not-withstanding any copyright annotation therein.


**References**


[1] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked
autoencoders are scalable vision learners. In _Proceedings of the IEEE/CVF Conference on_
_Computer Vision and Pattern Recognition_, pages 16000–16009, 2022.

[2] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In _European_
_conference on computer vision_, pages 776–794. Springer, 2020.

[3] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for
unsupervised visual representation learning. In _Proceedings of the IEEE/CVF conference on_
_computer vision and pattern recognition_, pages 9729–9738, 2020.

[4] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework
for contrastive learning of visual representations. In _International conference on machine_
_learning_, pages 1597–1607. PMLR, 2020.

[5] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena
Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar,
et al. Bootstrap your own latent-a new approach to self-supervised learning. _Advances in Neural_
_Information Processing Systems_, 33:21271–21284, 2020.

[6] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In _Proceedings_
_of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 15750–15758,
2021.

[7] Ashish Jaiswal, Ashwin Ramesh Babu, Mohammad Zaki Zadeh, Debapriya Banerjee, and Fillia
Makedon. A survey on contrastive self-supervised learning. _CoRR_, abs/2011.00362, 2020.

[8] Longlong Jing and Yingli Tian. Self-supervised visual feature learning with deep neural networks: A survey. _IEEE transactions on pattern analysis and machine intelligence_, 43(11):4037–
4058, 2020.

[9] Ishan Misra and Laurens van der Maaten. Self-supervised learning of pretext-invariant representations. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition_, pages 6707–6717, 2020.


10


[10] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin.
Unsupervised learning of visual features by contrasting cluster assignments. _Advances in Neural_
_Information Processing Systems_, 33:9912–9924, 2020.

[11] Yude Wang, Jie Zhang, Meina Kan, Shiguang Shan, and Xilin Chen. Self-supervised equivariant
attention mechanism for weakly supervised semantic segmentation. In _Proceedings of the_
_IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pages 12275–12284,
2020.

[12] Yann Dubois, Benjamin Bloem-Reddy, Karen Ullrich, and Chris J Maddison. Lossy compression
for lossless prediction. _Advances in Neural Information Processing Systems_, 34, 2021.

[13] Alexander Kolesnikov, Xiaohua Zhai, and Lucas Beyer. Revisiting self-supervised visual
representation learning. In _Proceedings of the IEEE/CVF conference on computer vision and_
_pattern recognition_, pages 1920–1929, 2019.

[14] Zeyu Feng, Chang Xu, and Dacheng Tao. Self-supervised representation learning by rotation
feature decoupling. In _Proceedings of the IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pages 10364–10374, 2019.

[15] Soroush Abbasi Koohpayegani, Ajinkya Tejankar, and Hamed Pirsiavash. Compress: Selfsupervised learning by compressing representations. _Advances in Neural Information Processing_
_Systems_, 33:12980–12992, 2020.

[16] Abhishek Sinha, Jiaming Song, Chenlin Meng, and Stefano Ermon. D2c: Diffusion-decoding
models for few-shot conditional generation. _Advances in Neural Information Processing_
_Systems_, 34:12533–12548, 2021.

[17] Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the
world. In _CVPR_, 2018.

[18] Gencer Sumbul, Marcela Charfuelan, Begüm Demir, and Volker Markl. Bigearthnet: A largescale benchmark archive for remote sensing image understanding. In _IGARSS 2019-2019 IEEE_
_International Geoscience and Remote Sensing Symposium_, pages 5901–5904. IEEE, 2019.

[19] Darius Lam, Richard Kuzma, Kevin McGee, Samuel Dooley, Michael Laielli, Matthew Klaric,
Yaroslav Bulatov, and Brendan McCord. xview: Objects in context in overhead imagery. _arXiv_
_preprint arXiv:1802.07856_, 2018.

[20] Adam Van Etten, Dave Lindenbaum, and Todd M Bacastow. Spacenet: A remote sensing
dataset and challenge series. _arXiv preprint arXiv:1807.01232_, 2018.

[21] Marshall Burke, Anne Driscoll, David B Lobell, and Stefano Ermon. Using satellite imagery to
understand and promote sustainable development. _Science_, 371(6535):eabe8628, 2021.

[22] Kumar Ayush, Burak Uzkent, Kumar Tanmay, Marshall Burke, David Lobell, and Stefano
Ermon. Efficient poverty mapping from high resolution remote sensing images. In _Proc. AAAI_
_Conf. Artif. Intell_, volume 35, pages 12–20, 2021.

[23] Kumar Ayush, Burak Uzkent, Marshall Burke, David Lobell, and Stefano Ermon. Generating interpretable poverty maps using object detection in satellite images. _arXiv preprint_
_arXiv:2002.01612_, 2020.

[24] Neal Jean, Marshall Burke, Michael Xie, W Matthew Davis, David B Lobell, and Stefano Ermon.
Combining satellite imagery and machine learning to predict poverty. _Science_, 353(6301):790–
794, 2016.

[25] Jiaxuan You, Xiaocheng Li, Melvin Low, David Lobell, and Stefano Ermon. Deep gaussian
process for crop yield prediction based on remote sensing data. In _Thirty-First AAAI conference_
_on artificial intelligence_, 2017.

[26] Anna X Wang, Caelin Tran, Nikhil Desai, David Lobell, and Stefano Ermon. Deep transfer
learning for crop yield prediction with remote sensing data. In _Proceedings of the 1st ACM_
_SIGCAS Conference on Computing and Sustainable Societies_, pages 1–5, 2018.

[27] Marc Rußwurm and Marco Körner. Self-attention for raw optical satellite time series classification. _ISPRS journal of photogrammetry and remote sensing_, 169:421–435, 2020.

[28] Jorge Andres Chamorro Martinez, Laura Elena Cué La Rosa, Raul Queiroz Feitosa,
Ieda Del’Arco Sanches, and Patrick Nigri Happ. Fully convolutional recurrent networks for multidate crop recognition from multitemporal image sequences. _ISPRS Journal of Photogrammetry_
_and Remote Sensing_, 171:188–201, 2021.


11


[29] Rose M Rustowicz, Robin Cheong, Lijing Wang, Stefano Ermon, Marshall Burke, and David
Lobell. Semantic segmentation of crop type in africa: A novel dataset and analysis of deep
learning methods. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition Workshops_, pages 75–82, 2019.


[30] Christopher Yeh, Chenlin Meng, Sherrie Wang, Anne Driscoll, Erik Rozi, Patrick Liu, Jihyeon Lee, Marshall Burke, David B Lobell, and Stefano Ermon. Sustainbench: Benchmarks
for monitoring the sustainable development goals with machine learning. _arXiv preprint_
_arXiv:2111.04724_, 2021.


[31] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In _2009 IEEE Conference on Computer Vision and Pattern_
_Recognition_, pages 248–255, 2009.


[32] Yutong He, Dingjie Wang, Nicholas Lai, William Zhang, Chenlin Meng, Marshall Burke,
David Lobell, and Stefano Ermon. Spatial-temporal super-resolution of satellite imagery via
conditional pixel synthesis. _Advances in Neural Information Processing Systems_, 34:27903–
27915, 2021.


[33] Yutong He, William Zhang, Chenlin Meng, Marshall Burke, David B Lobell, and Stefano
Ermon. Tracking urbanization in developing regions with remote sensing spatial-temporal
super-resolution. _arXiv preprint arXiv:2204.01736_, 2022.


[34] Kumar Ayush, Burak Uzkent, Chenlin Meng, Kumar Tanmay, Marshall Burke, David Lobell,
and Stefano Ermon. Geography-aware self-supervised learning. In _Proceedings of the IEEE/CVF_
_International Conference on Computer Vision_, pages 10181–10190, 2021.


[35] Oscar Mañas, Alexandre Lacoste, Xavier Giro-i Nieto, David Vazquez, and Pau Rodriguez. Seasonal contrast: Unsupervised pre-training from uncurated remote sensing data. In _Proceedings_
_of the IEEE/CVF International Conference on Computer Vision_, pages 9414–9423, 2021.


[36] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly,
Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image
recognition at scale. _ICLR_, 2021.


[37] Marc Rußwurm and Marco Körner. Convolutional lstms for cloud-robust segmentation of
remote sensing imagery. _arXiv preprint arXiv:1811.02471_, 2018.


[38] Charlotte Pelletier, Geoffrey I Webb, and François Petitjean. Temporal convolutional neural
network for the classification of satellite image time series. _Remote Sensing_, 11(5):523, 2019.


[39] Elisa Kamir, François Waldner, and Zvi Hochman. Estimating wheat yields in australia using
climate records, satellite image time series and machine learning methods. _ISPRS Journal of_
_Photogrammetry and Remote Sensing_, 160:124–135, 2020.


[40] Raí A Schwalbert, Telmo Amado, Geomar Corassa, Luan Pierre Pott, PV Vara Prasad, and
Ignacio A Ciampitti. Satellite-based soybean yield forecast: Integrating machine learning and
weather data for improving crop yield prediction in southern brazil. _Agricultural and Forest_
_Meteorology_, 284:107886, 2020.


[41] Chenlin Meng, Enci Liu, Willie Neiswanger, Jiaming Song, Marshall Burke, David Lobell, and
Stefano Ermon. Is-count: Large-scale object counting from satellite images with covariatebased importance sampling. In _Proceedings of the AAAI Conference on Artificial Intelligence_,
volume 36, pages 12034–12042, 2022.


[42] Evan Sheehan, Chenlin Meng, Matthew Tan, Burak Uzkent, Neal Jean, Marshall Burke, David
Lobell, and Stefano Ermon. Predicting economic development using geolocated wikipedia
articles. In _Proceedings of the 25th ACM SIGKDD international conference on knowledge_
_discovery & data mining_, pages 2698–2706, 2019.


[43] Enci Liu, Chenlin Meng, Matthew Kolodner, Eun Jee Sung, Sihang Chen, Marshall Burke,
David Lobell, and Stefano Ermon. Building coverage estimation with low-resolution remote
sensing imagery. _arXiv preprint arXiv:2301.01449_, 2023.


[44] Burak Uzkent, Evan Sheehan, Chenlin Meng, Zhongyi Tang, Marshall Burke, David Lobell,
and Stefano Ermon. Learning to interpret satellite images using wikipedia. In _Proceedings of_
_the Twenty-Eighth International Joint Conference on Artificial Intelligence_, 2019.


12


[45] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-chun
Woo. Convolutional lstm network: A machine learning approach for precipitation nowcasting.
_Advances in neural information processing systems_, 28, 2015.


[46] Andrei Stoian, Vincent Poulain, Jordi Inglada, Victor Poughon, and Dawa Derksen. Land
cover maps production with high resolution satellite image time series and convolutional neural
networks: Adaptations and limits for operational systems. _Remote Sensing_, 11(17):1986, 2019.


[47] Xin Yang and CP Lo. Using a time series of satellite imagery to detect land use and land cover
changes in the atlanta, georgia metropolitan area. _International Journal of Remote Sensing_,
23(9):1775–1798, 2002.


[48] Vivien Sainte Fare Garnot and Loic Landrieu. Panoptic segmentation of satellite image time
series with convolutional temporal attention networks. In _Proceedings of the IEEE/CVF_
_International Conference on Computer Vision_, pages 4872–4881, 2021.


[49] Aidan M Swope, Xander H Rudelis, and Kyle T Story. Representation learning for remote
sensing: An unsupervised sensor fusion approach. _arXiv preprint arXiv:2108.05094_, 2021.


[50] Vladan Stojnic and Vladimir Risojevic. Self-supervised learning of remote sensing scene
representations using contrastive multiview coding. In _Proceedings of the IEEE/CVF Conference_
_on Computer Vision and Pattern Recognition_, pages 1182–1191, 2021.


[51] Pallavi Jain, Bianca Schoen-Phelan, and Robert Ross. Multi-modal self-supervised representation learning for earth observation. In _2021 IEEE International Geoscience and Remote Sensing_
_Symposium IGARSS_, pages 3241–3244. IEEE, 2021.


[52] Wenyuan Li, Hao Chen, and Zhenwei Shi. Semantic segmentation of remote sensing images
with self-supervised multitask representation learning. _IEEE Journal of Selected Topics in_
_Applied Earth Observations and Remote Sensing_, 14:6438–6450, 2021.


[53] Wenyuan Li, Keyan Chen, Hao Chen, and Zhenwei Shi. Geographical knowledge-driven
representation learning for remote sensing images. _IEEE Transactions on Geoscience and_
_Remote Sensing_, 2021.


[54] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are
data-efficient learners for self-supervised video pre-training. _arXiv preprint arXiv:2203.12602_,
2022.


[55] Hongxu Chen, Sixiao Zhang, and Guandong Xu. Graph masked autoencoder. _arXiv preprint_
_arXiv:2202.08391_, 2022.


[56] Roman Bachmann, David Mizrahi, Andrei Atanov, and Amir Zamir. Multimae: Multi-modal
multi-task masked autoencoders. _arXiv preprint arXiv:2204.01678_, 2022.


[57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information_
_processing systems_, 30, 2017.


[58] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and
Hervé Jégou. Training data-efficient image transformers & distillation through attention. In
_International Conference on Machine Learning_, pages 10347–10357. PMLR, 2021.


[59] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luˇci´c, and Cordelia
Schmid. Vivit: A video vision transformer. In _International Conference on Computer Vision_
_(ICCV)_, 2021.


[60] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin
transformer. _arXiv preprint arXiv:2106.13230_, 2021.


[61] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_,
pages 770–778, 2016.


[62] Xinlei Chen*, Saining Xie*, and Kaiming He. An empirical study of training self-supervised
vision transformers. _arXiv preprint arXiv:2104.02057_, 2021.


[63] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel
dataset and deep learning benchmark for land use and land cover classification. _IEEE Journal_
_of Selected Topics in Applied Earth Observations and Remote Sensing_, 12(7):2217–2226, 2019.


13


[64] Maxim Neumann, Andre Susano Pinto, Xiaohua Zhai, and Neil Houlsby. In-domain representation learning for remote sensing. _arXiv preprint arXiv:1911.06721_, 2019.

[65] Hengshuang Zhao, Yi Zhang, Shu Liu, Jianping Shi, Chen Change Loy, Dahua Lin, and Jiaya
Jia. Psanet: Point-wise spatial attention network for scene parsing. In _Proceedings of the_
_European conference on computer vision (ECCV)_, pages 267–283, 2018.

[66] Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres. Quantifying
the carbon emissions of machine learning. _CoRR_, abs/1910.09700, 2019.


14


**A** **Appendix**


**A.1** **Datasets**


**fMoW RGB** Functional Map of the World (fMoW) [ 17 ] is a dataset of high-resolution satellite
image time series across the world, with a task of classification among 62 architecture categories such
as airport, shipyard, and zoo. fMoW provides RGB images as well as metadata including location,
time, sun angles, etc. The license is provided here [2] .


Co-located images of different timestamps, or sequences, are provided in fMoW. They are of different
length, and around 60% of the samples have length larger than 2. Readers can refer to the fMoW
paper [ 17 ] for statistics on the distribution of sequence lengths. We construct a temporal version
of fMoW by randomly associating every single image with two images of the same location but of
different timestamps if possible. For a given spatial location _loc_, we define _T_ _loc_ as the number of
temporally distinct snapshots present in the dataset.


**fMoW Sentinel** We collect a new dataset based on the fMoW RGB dataset. We crop surface
reflectance images from the Sentinel-2 (ESA) satellite (courtesy of the U.S. Geological Survey),
consisting of 90-day composites of images at the same locations as fMoW images (to reduce the
impacts of cloud coverage). At each fMoW datapoint location, we collect a time series of Sentinel-2
images, using the provided geo-coordinate bounding boxes. For locations where all fMoW images are
before the Sentinel-2 time range, we discard the location. Otherwise, we collect a composite centered
at the same time as each fMoW image within Sentinel-2 time range. Because many fMoW images
occur before Sentinel-2, we augment the time series by adding extra images in 6-month intervals that
do not have an image in the fMoW dataset.


We collect all 13 frequency bands provided by Sentinel-2 (B1-12 and B8A), at some of the same times
as fMoW images plus some extra times, for a total of 712,874 training images, 84,939 validation
images, and 84,966 test images. Out of these 155,446 training images, 22,602 validation images, and
22,824 test images occur at the same time as a corresponding fMoW image. The mean height and
width of each image is about 45 pixels and 60 pixels, respectively.


Figure 5: Distribution of images and locations across the categories over the fMoW Sentinel training set.


**A.2** **fMoW Sentinel**


We provide information about the fMoW-Sentinel dataset, collected using Sentinel-2 [3] .


2 fMoW license: `[https://github.com/fMoW/dataset/raw/master/LICENSE](https://github.com/fMoW/dataset/raw/master/LICENSE)`
3 Sentinel-2 license: `[https://scihub.copernicus.eu/twiki/pub/SciHubWebPortal/](https://scihub.copernicus.eu/twiki/pub/SciHubWebPortal/TermsConditions/Sentinel_Data_Terms_and_Conditions.pdf)`
```
TermsConditions/Sentinel_Data_Terms_and_Conditions.pdf

```

15


**A.2.1** **Geographic Distribution**


Figure 6: Geographic distribution of fMoW Sentinel images by country


**A.2.2** **fMoW Sentinel Bands**

|Channel|Resolution Central wavelength Mean Standard deviation|
|---|---|
|B1: Aerosols<br>B2: Blue<br>B3: Green<br>B4: Red<br>B5: Red Edge 1<br>B6: Red Edge 2<br>B7: Red Edge 3<br>B8: NIR<br>B8A: Red Edge 4<br>B9: Water Vapor<br>B10: Cirrus<br>B11: SWIR 1<br>B12: SWIR 2|60m<br>443nm<br>1370.192<br>633.152<br>10m<br>490nm<br>1184.382<br>650.284<br>10m<br>560nm<br>1120.771<br>965.231<br>10m<br>665nm<br>1136.260<br>948.982<br>20m<br>705nm<br>1263.739<br>1108.067<br>20m<br>740nm<br>1645.403<br>1258.364<br>20m<br>783nm<br>1846.870<br>1233.149<br>10m<br>842nm<br>1762.595<br>1364.387<br>20m<br>865nm<br>1972.624<br>3545.66<br>60m<br>940nm<br>582.726<br>472.380<br>60m<br>1375nm<br>14.771<br>14.311<br>20m<br>1610nm<br>1732.164<br>1310.370<br>20m<br>2190nm<br>1247.919<br>1087.602|



Table 10: Mean and standard deviation of pixel values for each channel across the fMoW Sentinel training
dataset. Note that channel B10 does not contain bottom-of-atmosphere information, and is no longer accessible
[on Google Earth Engine. Further details can be found here.](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial)


Figure 7: Distribution of pixel counts per band across the fMoW Sentinel training set.


16


**A.3** **Training Details**


Here we describe the settings used for pre-training and finetuning our models on fMoW RGB (nontemporal) (A.3.1), fMoW RGB (temporal) (A.3.2), fMoW Sentinel (A.3.3), NAIP (A.7), EuroSAT
(A.8), BigEarthNet (A.9), and SpaceNet v1 (A.10).


**A.3.1** **fMoW RGB (non-temporal)**


**SatMAE Pre-training** We use ViT-Large [ 36 ] as the backbone. The model configuration is the
same as in [ 1 ], e.g. the input image size is 224 and the patch size is _P_ = 16 . Since the original image
size of fMoW varies greatly, we first resize the image so that the shorter side is 224 pixels and the
aspect ratio is maintained, then randomly crop a 224 _×_ 224 region from the resized image. We also
normalize the image according to the mean and standard deviation calculated on the whole dataset.
We use 8 NVIDIA V100 GPUs on Google Cloud to train the model for 800 epochs with a learning
rate of 2 _._ 4 _×_ 10 _[−]_ [3] and batch size of 4096 . The optimizer and learning rate scheduler are kept the
same as in [1].


**SatMAE Finetuning** We load the pre-trained weights below the ViT head and finetune the ViTLarge model in an end-to-end manner. We adopt the same learning rate decay and weight decay
strategy during finetuning as in [ 1 ]. We apply the same data augmentation during pre-training and
additionally use Mixup and Cutmix augmentation. We use 8 NVIDIA V100 GPUs to train the model
for 50 epochs with a learning rate of 2 _×_ 10 _[−]_ [3] and batch size of 512 . Other paramters including
Mixup coefficients are kept the same as in [1].


**A.3.2** **fMoW RGB (temporal)**


**Dataset** We iterate over the dataset in the same way as in non-temporal fMoW, except that we
randomly find 2 co-located images with different timestamps (if possible) for every image sample,
so every sample becomes a image sequence of length 3. If there are not enough co-located images
with different timestamps we simply duplicate the original image. The fMoW dataset train/val split
guarantees that co-located images belong to the same split so there is no leakage involved.


**SatMAE Pre-training** We use the same model as above, though the number of input patches triples.
To incorporate temporal encoding, the positional encoding of a spatial location of a patch shortens to a
320 + 320 = 640 dimensional vector, and the temporal encoding is a 384 dimensional vector, divided
equally among the year, month, and hour. We apply constraints on the mask indices to implement
different mask strategies. For _independent masking_ (4.1.2), we pick the variant where we keep the
ratio of masked patches fixed to _p_ _m_ = 0 _._ 75 for each image in the sequence. For the consistent
cropping option, we first resize the image sequence to the same size and then apply cropping to all
three images instead of randomly cropping each image separately. We use 8 NVIDIA V100 GPUs to
train the model for 100 epochs with a learning rate of 6 _×_ 10 _[−]_ [4] and batch size of 1024.


**SatMAE Finetuning** We use 4 or 8 NVIDIA V100 GPUs to train the model for 50 epochs with a
learning rate of 5 _×_ 10 _[−]_ [4] and batch size of 128.


**Test-time Augmentation** Unlike the test-time augmentation used in [ 34 ], we average the prediction
score of 9 random samples of image sequences for every single image as the final prediction score.
To be consistent with previous experiments, we calculate the mean classification accuracy on the
whole validation set instead of evaluating on the subset with unique locations. These two metrics
give very similar numbers.


**SeCo [35] Pre-training and Finetuning** We use the code from the official repo of SeCo [ 35 ], and
use ResNet 50 as the backbone. For pre-training, we use 8 NVIDIA v100 GPUs and a batch size of
128, and keep other hyper-parameters and data augmentation the same as in [ 35 ]. We pre-train the
model for 50 epochs and observe the loss converged. For finetuning, we use 4 NVIDIA v100 GPUs
and a batch size of 128, and also keep other hyper-parameters and data augmentation the same as in

[35]. We finetune the model for 100 epochs.


17


**UTAE [48] Training** We use the code from the official repo of UTAE [ 48 ], add an averaging
pooling layer to adapt the segmentation network to classification. We use 8 NVIDIA v100 GPUs, a
batch size of 128, learning rate of 5 _×_ 10 _[−]_ [4], and use AdamW optimizer with no weight decay, which
we found to be the best performing hyperparameters. We apply data augmentation the same as in
SatMAE. We train the model for 50 epochs.


**A.3.3** **fMoW Sentinel**


We choose the ViT-Large backbone [ 36 ] with _D_ = 1024 . The positional encoding of the spatial
location of a patch is a 768 dimensional vector, and the spectral group encoding is a 256 dimensional
vector. Given the relatively smaller size of Sentinel-2 imagery, we resize all images to 96 _×_ 96
pixels and use a patch size _P_ = 8 . This results in _L_ = (96 _/_ 8) [2] = 144 patches which are passed
to SatMAE+Stack. For SatMAE+Group, since we pick 3 groups of channels, we have 3 _L_ = 432
patches. We did experiment with letting each channel be its own group. However, this resulted in a
very large memory footprint with 10 _L_ = 1440 patches and unstable training which would frequently
result in NaN loss. We thus decided to group bands in terms of spatial resolution and wavelength
similarity (see 5.4). We train and finetune on the entire training set.


**SatMAE Pre-training** We use 8 NVIDIA v100 GPUs, an effective batch size of 4096, a base
learning rate of 10 _[−]_ [4] and the same warmup and half-cyle cosine decay schedule used by [ 1 ]. For each
image, we use standard normalisation (see statistics in A.2.2), randomly crop 0 _._ 2 - 1 _._ 0 _×_ of the area of
the image, resize it to 96 _×_ 96 pixels, and randomly flip the image horizontally. We use a masking
ratio of _p_ _m_ = 0 _._ 75, as was found to be optimal in [1]. We pre-train each model for 50 epochs.


**MoCo Pre-training** We use 8 NVIDIA v100 GPUs, and pick the ViT-Base backbone and an
effective batch size of 512 such that the model fits in memory. We pick a base learning rate of
10 _[−]_ [4] and the same warmup and decay schedule used by [ 62 ]. For each image, we use standard
normalisation (see statistics in A.2.2), randomly crop 0.2-1.0x of the area of the image, resize it
to 96 _×_ 96 pixels, randomly apply Gaussian blur with _σ ∈_ [0 _._ 1 _,_ 2], and randomly flip the image
horizontally. We pre-train each model for 50 epochs and use the 50th epoch checkpoint for all
subsequent experiments.


**SatMAE Finetuning** We use 8 NVIDIA v100 GPUs, an effective batch size of 4096, a base
learning rate of 10 _[−]_ [3] and a warmup and decay schedule. We use standard normalisation, and resize
each image to 96 _×_ 96 pixels.


**SatMAE Further Improvements** We found increased performance of around 2.18% during finetuning (the last row of table 3) using additional data augmentations as in [ 1 ]. This configuration is
the same as SatMAE+Group+IM except for an effective batch size of 1024, weight decay of 0.05,
drop path of 0.1, reprob of 0.25, mixup of 0.8 and cutmix of 1.0. For all rows, we finetune for 30
epochs, but report results on the best validation set Top 1 accuracy achieved.


In table 11, we also report results with increased

Backbone Pre-train epochs Top 1 Acc. pre-training. Training SatMAE for 200 epochs, as
ViT-B 200 62.65 opposed to 50, yields further improvements in the

final top 1 accuracy after finetuning for 30 epochs

ViT-L 50 61.48

using the configuration described above. We see

ViT-L 200 **63.84**

that a smaller model, using a ViT-Base backbone,

Table 11: Improvements with longer pre-training. can outperform a model using a ViT-Large back
bone with longer pre-training. We hypothesize that
longer pre-training can prove to be even more beneficial.



Backbone Pre-train epochs Top 1 Acc.
ViT-B 200 62.65

ViT-L 50 61.48

ViT-L 200 **63.84**



Table 11: Improvements with longer pre-training.



**A.4** **Impact of masking ratio and patch size on fMoW-RGB-temporal**


Here, we investigate the impact of the masking ratio and patch size for a ViT-Large SatMAE
on temporal data (with independent masking and consistent cropping, and without the test time
augmentation).


We vary the masking ratio _p_ _m_ to 0.6 and 0.9 from a default of _p_ _m_ = 0 _._ 75 and the patch size _P_ to 22,
32 from a default of _P_ = 16 (table 12).


18


_p_ _m_ _P_ Top 1 Acc.
0.6 16 72.80

0.9 16 74.78

0.75 32 69.31

0.75 22 72.08

0.75 16 **79.69**



We see a significant drop in performance of 7 _._ 19% with a smaller
masking ratio as expected [1], since a lower masking ratio makes it
easier for the model to reconstruct masked patches as it has access
to more visible patches. A higher masking ratio of 0 _._ 9 may result in
a difficult pretext task, as too few of the patches in the image remain
visible, which may require longer training. Thus, we find that using
_p_ _m_ = 0 _._ 75 is roughly optimal.



Table 12: Ablation on _p_ _m_ and _P_ We also note a drop in performance from using a larger patch size,
on fMoW-RGB-temporal.

as the model has access to less granular spatial information from
the image. This is in line with other MAE works [ 1, 54 ]. However,
using a larger patch size is also more computationally efficient, so one must consider the tradeoff in
accuracy and computational resources.


**A.5** **Impact of masking ratio and patch size on fMoW-Sentinel**



_p_ _m_ _P_ Top 1 Acc.
0.6 8 50.68

0.9 8 59.28

0.75 16 55.02

0.75 8 **59.30**



In this section, we investigate the impact of the masking ratio _p_ _m_
and the patch size _P_ (table 13) for SatMAE on multi-spectral data.
We use a ViT-Large backbone and the SatMAE+Group+IM setting
as this was our best performing design.



0.75 16 55.02 As we expect, a lower masking ratio results in a weak pretext task, as
0.75 8 **59.30** the model is able to easily reconstruct the image given more visible

patches, and thus its representations are not as useful. Interestingly,

Table 13: Ablation on _p_ _m_ and _P_ _p_ _m_ = 0 _._ 9 doesn’t result in a large drop in performance, unlike [ 1 ].
on fMoW-Sentinel.

This suggests that higher masking ratios may be used for multispectral data with independent masking, as it results in fewer tokens
during the encoding state which could quicken pre-training.



Table 13: Ablation on _p_ _m_ and _P_
on fMoW-Sentinel.



We see that a larger patch size results in worse performance. This is expected, as a larger patch size
provides less granular spatial information to the deeper layers of the model, which may dampen its
expressive power. As mentioned above, the loss in accuracy must be considered compared against
the gain in training speed. A future direction of research could consider the specific gain in speed and
drop in accuracy from granular increases to the patch size for further insight.


**A.6** **Choosing important multi-spectral bands**


As mentioned in 5.4, all 13 bands of the Sentinel-2 data may not be useful. In our experiments,
we drop bands B1, B9, and B10. To correctly identify the utility of each band, one would need to
pre-train a model with all bands except the one in question, and then measure the performance after
finetuning without that band. However, this is prohibitively expensive in terms of computational
resources. Instead, we pre-trained a SatMAE+Stack model with a ViT Base backbone on all 13
Sentinel-2 bands of fMoW Sentinel, and then finetuned the model on the image classification task of
fMoW Sentinel using all 13 bands. Using the finetuned model, we ran an ablation masking out subsets
of bands with the mean value for those bands and measuring the drop in validation set accuracy. Since
the model was trained to rely on information of all 13 bands, a small drop in accuracy from masking
out some bands indicates that these bands might not be very useful for the model to perform well.


Bands Masked Top 1 Acc. Top 5 Acc.
None 57.80 80.07

B1 45.83 69.46

B2, B3, B4 17.46 35.05
B5, B6, B7 13.25 35.33
B8, B8A 16.03 35.36
B9, B10 55.06 78.15
B11, B12 27.83 54.41

Table 14: An ablation to determine which of the 13 bands are least useful to a SatMAE+Stack model pre-trained
and finetuned on all 13 bands of fMoW Sentinel. During evaluation, for each image, the relevant bands are
masked with their mean value recorded in table 10 and then passed as is to the finetuned SatMAE+Stack model.


19


As seen in table 14, we notice the smallest drop in accuracy when masking bands B9 and B10. The
drop in accuracy when masking B1 is larger, but could be due to the model relying on potentially
unimportant signals from B1 during finetuning. We therefore also drop B1 in our experiments in
section 5.4. We note that the RGB and other multi-spectral bands are highly relevant to our model.


**A.7** **NAIP Land Cover Classification**


We use the finetuning setting as in the fMoW RGB (non-temporal) finetuning experiment (A.3.1).


**A.8** **EuroSAT Land Cover Classification**


We use the exact finetuning setting as in the fMoW RGB (non-temporal) finetuning experiment for
RGB-only input (A.3.1). We also use the exact finetuning setting as in the fMoW-Sentinel finetuning
experiment (A.3.3) except for training longer (150 epochs) for multi-spectral (13-band) input. It
took the model longer to converge most probably because EuroSAT is comparatively a much smaller
dataset. The license [4] is provided in the footnote.


**A.9** **BigEarthNet Land Cover Multi-label Classification**


We use the exact finetuning setting as in the fMoW-Sentinel finetuning experiment (A.3.3). Since
the task is multi-label classification instead of single-label classification, we changed the training
objective to multi-label soft margin loss. We use the mean Average Precision metric as provided in

[35].The license [5] is provided in the footnote.


**A.10** **SpaceNet v1 Building Segmentation**


We use PSANet [ 65 ] for the binary image segmentation and replace the backbone with ViT-Large.
Following [ 34 ], we set the learning rate to 1 _×_ 10 _[−]_ [3] for ViT encoder and to 1 _×_ 10 _[−]_ [2] for ViT head
and PSA module. We train the model for 100 epochs with batch size 128 using an SGD optimizer
of momentum 0 _._ 9 and weight decay 1 _×_ 10 _[−]_ [4] and a polynomial learning rate decay scheduler of
power 0 _._ 9 . Also following [ 34 ], we resize and crop the input image to 400 _×_ 400 for fair comparison.
This indicates our model will take more patches per image (625). We use the positional encoding
interpolation algorithm provided by [ 1 ] to adjust the pre-trained weights. The license [6] is provided in
the footnote.


4 EuroSAT license: `[https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)`
5 BigEarthNet license: `https://bigearth.net/downloads/documents/License.pdf`
6 SpaceNet v1 license: `http://creativecommons.org/licenses/by-sa/4.0/`


20


**B** **Societal Impact**


Measurements of economic, social, and environmental indicators are critical to policy-making across
the world. However, such measurements are constantly lacking, hindering the process of decision
making. Instead of using traditional measurements ( _e.g_ . ground survey), our method exploits abundant, globally-available and frequently-updated unlabelled satellite data. Our model is capable of
capturing representations from remote sensing imagery that are beneficial for critical downstream
tasks, including poverty prediction, infrastructure development, and population estimation. Governments could make good use of such information in decision making and consequently bring
significant societal benefits.


Although the use of satellite imagery could potentially lead to data abuse and privacy violations from
malicious actors, we contend that applications of our model trained on publicly available satellite
imagery respects privacy and avoids exposing sensitive information. For example, individually
identifiable information cannot easily be retrieved from these images. Thus, we believe that the
imagery we used does not directly constitute a privacy concern. However, we note that representations
learned from SatMAE could potentially suffer from biases if the training data is biased. For instance,
SatMAE trained on geographically imbalanced data could bias the model towards certain regions,
especially those in Northern America and Europe (see 6). Thus, we advise researchers to be aware of
directly applying our SatMAE models to datasets with a geographical distribution different to that
of fMoW RGB and fMoW Sentinel. In our code release, we will also specify allowable uses with
appropriate licenses.


**B.1** **Carbon Footprint**


We include a brief analysis of the carbon footprint of training the model below.


Our experiments were mainly conducted using Google Cloud Platform (GCP) in region us-central1,
which has a carbon efficiency of 0.57 kg _CO_ 2 eq. per kWh. For a model pre-trained and finetuned on
fMoW RGB (temporal) dataset, a cumulative of 960 hours of computation was required on hardware
of type Tesla V100-SXM2-16GB (TDP of 250W). Total emissions are estimated to be 136.8 kg _CO_ 2
eq. of which 100 percent was directly offset by the cloud provider. Estimations were conducted using
the Machine Learning Impact calculator presented in [ 66 ]. For a model pre-trained and finetuned on
fMoW Sentinel dataset, total emissions are estimated to be 109.44 kg _CO_ 2 eq. We list a table for the
rough estimations in table 15.


Experiment Carbon Footprint
Dataset GPU hours
Setting (kg _CO_ 2 eq.)
Pre-training fMoW RGB temporal 768 109.44
Finetuning fMoW RGB temporal 192 27.36
Pre-training fMoW Sentinel 576 82.08
Finetuning fMoW Sentinel 192 27.36
Finetuning NAIP 30 4.27
Finetuning EuroSAT 4 0.57
Finetuning SpaceNet 50 7.12
Finetuning BigEarthNet 16 2.28

Table 15: The estimated carbon footprint of pre-training and finetuning SatMAE on these datasets. The GPU
hours are measured on 8 NVIDIA v100 GPUs in the us-central1 region on GCP.


21


**C** **Visualizations**


In this section, we include visualisations in the temporal (C.1) and multi-spectral settings (C.2).


**C.1** **Temporal SatMAE**


Figure 8: More visualization examples of the reconstruction quality of image sequences from the fMoW temporal
dataset across multiple settings, including using no temporal encoding + IM (Independent Masking), default +
CM, default + IM.


As shown in fig. 8, SatMAE+IM achieved relatively satisfying reconstruction quality. Without the
temporal encoding, the patches across all three images cannot be distinguished, and we observe
a mixture of the three images in the reconstruction outcome of the second column. As explained
earlier, using independent masking can allow SatMAE to reconstruct an image in the time series
using information from other temporal patches. Our experiments show that this helps SatMAE learn
better representations for satellite imagery.


**C.2** **Spectral SatMAE**


We also visualize the in-painting quality for different multi-spectral settings, including SatMAE+Group+IM, SatMAE+Group+CM (4.2, 4.2), and SatMAE+Stack (4.2) in fig. 9 and fig. 10.


We can see a clear improvement in the quality of reconstruction under SatMAE+Group+IM compared
to SatMAE+Group+CM and SatMAE+Stack. Independent masking results in sharper reconstructions,
whereas the results from consistent masking and stacking the channels are much fuzzier. We also


22


Figure 9: Visualizations of SatMAE in-painting under different settings. RGB represents bands B4, B3, B2,
from group B2, B3, B4, B8. NIR represents bands B7, B6, B5, from group B5, B6, B7, B8A. SWIR represents
bands B11 in grayscale, from group B11, B12. For each method, we show the input band group masked and
reconstructed side-by-side. We note that the reconstruction for visible patches is worse than for the masked
patches, since no loss is computed on visible patches. Both halves represent multi-spectral images of airports.


note that the model is able to learn correlations between bands; in the top-half of fig. 9 for the SWIR
band group, even though the bottom right corner of the image is masked, SatMAE+Group+IM is able
to reconstruct the bright spot based on information from the other band groups.


We hypothesize that further improvements in reconstruction quality and learned representations can
be achieved with longer pre-training.


23


Figure 10: Further visualizations of SatMAE in-painting. See fig. 9 for details on band groups. The top half
represents a multi-spectral image of a recreational facility and the bottom half is of an amusement park.


24


