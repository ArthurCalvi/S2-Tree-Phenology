## **Evaluating and Benchmarking Foundation** **Models for Earth Observation and Geospatial AI**

Nikolaos Dionelis [1], Casper Fibaek [1], Luke Camilleri [1] _[,]_ [2], Andreas Luyts [1] _[,]_ [3], Jente
Bosmans [1], Bertrand Le Saux [1]


1 European Space Agency (ESA), ESRIN, _Φ_ -lab, Italy, 2 Trust Stamp, 3 VITO


**Abstract.** When we are primarily interested in solving several problems jointly with a given prescribed high performance accuracy for each
target application, then Foundation Models should be used rather than
problem-specific models. We focus on the specific vision application of
Foundation Models for Earth Observation (EO) and geospatial AI. These
models can solve important problems we are tackling, including for example land cover classification, crop type mapping, flood segmentation,
building density estimation, and road regression segmentation. In this
paper, we show that for a limited number of labelled data, Foundation Models achieve improved performance compared to problem-specific
models. In this work, we also present our proposed evaluation benchmark for Foundation Models for EO. Benchmarking the generalization
performance of Foundation Models is important as it has become difficult to standardize a fair comparison across the many different models.
We present the results using our evaluation benchmark for EO Foundation Models and show that Foundation Models are label efficient in the
downstream tasks and help us solve problems we are tackling in EO.


**Keywords:** Foundation Models for Earth monitoring · Evaluation bench.


**1** **Introduction**


An advantage of Foundation Models compared to problem-specific models is that
for a limited number of labelled data, Foundation Models achieve improved performance. Label efficiency is important in real-world applications as for many use
cases, both labelling and continuous re-labelling are needed. In the specific case
of Earth Observation (EO) and remote sensing, labels _change_ over time. Also,
data from satellites are _unlabelled_ . Annotating such data is difficult, requires
expertise, and is costly in terms of time. An additional advantage of Foundation
Models is that they perform _sharing_ across tasks and learn a common module,
for example segmentation, needed for all the target applications we are trying to
solve jointly with a given prescribed high performance accuracy for _each_ task.
The target applications of EO Foundation Models are important problems
we are trying to _solve_, such as land cover classification semantic segmentation,
crop type mapping, and crop yield estimation. Additional target applications
are flood segmentation, building density estimation, road _regression_ segmentation, estimation of the age of buildings, marine litter detection, methane plume


2 N. Dionelis, C. Fibaek, et al., Submitted


segmentation, and change detection for wildfires, floods, and anomalies. Furthermore, there are also important EO problems that we would like to solve for
which we have _only_ unlabelled data, i.e. no labels, for example iceberg detection.


**2** **Solving** _**M**_ **tasks jointly with prescribed high accuracy**


Given a prescribed high performance for each task, e.g., accuracy 95%, we deal
with _M_ problems _jointly_ . For EO Foundation Models, we address approximately
_M_ = 10 target applications together. The prescribed high performance is crucial
as we want the model to be useful; otherwise, people will _not_ use it. For Earth
monitoring, we want generalization to a big geographical area/ large inference
set. The performance stringent requirement drives everything. The _two_ alternatives are the following. For the use cases, for datasets D1, D2, ..., D _M_ that have
labels, the alternative A is to perform supervised learning on the datasets. We
name these tasks P1, P2, ..., P _M_ . The alternative B is to perform _self-supervised_
learning on a common dataset _D_ _[′]_ . We name this task _L_ . Then, we perform supervised learning for the target applications. We name these tasks Q1, Q2, ...,
Q _M_ . The dataset _D_ _[′]_ contains relevant data, e.g., similar objects or data from
the same satellite. The alternative A is using problem-specific models, solving
_each_ problem on its own, and assuming the existence of a _lot_ of labels for each
use case. The alternative B is using a common model and solving _groups_ of tasks
that are of interest to us. Big common/ shared models are Foundation Models.
For the alternative A, problem-specific models do _not_ have label efficiency: for
limited labelled data, they yield _low_ performance accuracy (or F1-score or Intersection over Union (IoU)). There is no sample efficiency for these models and
we have to pay too much and _wait_ too long for the labels. The performance
requirement drives everything as the data size mainly depends on the prescribed
high accuracy. The relationship between the size of the data and the accuracy
is approximately linear. We cannot escape the _large_ size of the dataset because
of the performance stringent requirement. In EO, the data size is: some TBs.
Using common/ shared models is _beneficial_ : we learn the common representations. There is sharing across tasks: we learn the commonality, i.e. the common
operations (segmentation) for _re-usability_ and efficiency. For the alternative B,
i.e. for common models and Foundation Models, _N_ % of the labels are needed
that would otherwise be required. For EO Foundation Models, _N ≈_ 20 and _even_
10. For the alternative A (problem-specific models), _all_ the labels are needed.
For the alternative A, the cost C1 (which is also directly related to the data
_size_ and how large the architecture needs to be as these three are _similar_ ) is:


C1 = P1 + P2 + _..._ + P _M ≈_ _My_, (1)


where typically _M_ = 10 tasks and _y_ is the cost or data for one task. Because of
the _high_ accuracy requirement, _y_ is large, e.g., 100000. This is why for problemspecific models, the cost, as well as the data size and how large is the architecture,
is _times_ the number of tasks. For _M_ = 10 use cases, for the alternative A, we


Evaluating and Benchmarking Foundation Models for Earth Observation 3


have times 10, i.e. C1 = 10 _y_ from (1). Next, for the alternative B, the cost is:


C2 = _L_ + Q1 + Q2 + _..._ + Q _M ≈_ _y_ + _N_ % _y M_ = _y_ (1 + _N_ % _M_ ). (2)


This scales better than C1, i.e. C2 = 3 _y_ . Overall, C2 _≈_ 300000, C1 _≈_ 1 _M_, and
C2 _<_ C1. Big common models achieve label efficiency for both segments and semantics. Segment label efficiency refers to the segments and their _shape_ . For both
segment and semantic label efficiency, in remote sensing, continuous re-labelling
is needed as we live in a _dynamic_ world: Earth is ever-changing. Human annotators are needed, as well as expert knowledge. Also, imperfect labels exist in EO,
i.e. _noisy_ labels. C1 grows linearly with _M_, i.e. O( _M_ ), while C2 _grows_ linearly
with _N_ % _M_, i.e. O( _N_ % _M_ ). Because of the accuracy requirement and the _linear_
relationship between the data size and the accuracy, for problem-specific models,
we train 10 models that are _approximately_ as large as 10 Foundation Models,
i.e. it is like training 10 Foundation Models. Also, for problem-specific models,
a lot of labels are needed which are expensive in terms of _both_ cost and time.


**3** **Our Proposed Evaluation Benchmark for FMs for EO**


Evaluating and benchmarking Foundation Models in terms of their generalization performance is important as it has become increasingly difficult to standardize a _fair_ comparison across the many different models. For the specific vision
application of Foundation Models for EO and geospatial AI [1,2,3], we present our
proposed evaluation benchmark and show that for a _limited_ number of labelled
data, Foundation Models achieve improved results compared to problem-specific
models. Foundation Models are label efficient in the downstream tasks [4,5]. For
semantic segmentation land cover classification (lc), the evaluation results are
presented in Fig. 1. We examine both _fine-tuning_ (ft) and linear probing (lp).


**Fig. 1.** Evaluating, _benchmarking_, and ranking Foundation Models for EO and geospatial AI on the downstream task of semantic segmentation land cover classification.


4 N. Dionelis, C. Fibaek, et al., Submitted


Geo-location classification pre-training is used for the models that we have developed in-house. These are the _geo-aware_ models in Fig. 1. As a pre-text task, our
Foundation Model Version 1.0 performs longitude and latitude satellite metadata information learning. For this, we have used a _global_ unlabelled dataset of
satellite Sentinel-2 L2A data and 10 spectral bands. As a downstream task, we
perform fine-tuning (or linear probing) on the labelled dataset WorldCover [1] . According to the results in Fig. 1, the percentage _improvement_ of Foundation Models compared to problem-specific models is approximately 18 _._ 52% when there are
limited samples of labelled data, e.g., 100 images per region (geo-aware U-Net
ft and U-Net fully-supervised). We have examined both a _Transformer_ -based
architecture, i.e. Vision Transformer (ViT), and a U-Net-based architecture.
For the task of estimating the label at the image level (rather than at the
_pixel_ level) for land cover classification, according to our results, the percentage
improvement of Foundation Models compared to problem-specific models is approximately 16 _._ 36% when limited labels are used, e.g., 100 samples per region
(geo-aware U-Net ft vs. U-Net fully-supervised, 0 _._ 64 and 0 _._ 55 respectively).
Next, for the task of estimating how dense and close to each other buildings
are, the results are presented in Fig. 2. For this _regression_ downstream task, the
evaluation metric is the Mean Squared Error (MSE). We compare 15 models in
total. For this specific use case, the percentage improvement of Foundation Models _compared_ to problem-specific models is 86% when there are limited labelled
data: 100 samples per region (geo-aware U-Net and U-Net fully-supervised).


**Fig. 2.** Evaluation of Foundation Models for EO on the target application of estimating
how dense and _close_ to each other buildings are, in the MSE metric (regression task).


**4** **Conclusion**


To solve several problems jointly with a prescribed high accuracy for each task,
we use Foundation Models. For the vision application of Foundation Models for
EO, for limited labelled data, Foundation Models outperform problem-specific
models in our proposed evaluation benchmark for Foundation Models for EO.


1 [http://worldcover2020.esa.int/data/docs/WorldCover_PUM_V1.1.pdf](http://worldcover2020.esa.int/data/docs/WorldCover_PUM_V1.1.pdf)


Evaluating and Benchmarking Foundation Models for Earth Observation 5


**References**


1. Jakubik, J., Roy, S., et al.: Foundation Models for Generalist Geospatial Artificial
[Intelligence, arXiv:2310.18660 (2023)](http://arxiv.org/abs/2310.18660)
2. Bastani, F., Wolters, P., et al.: Satlaspretrain: A large-scale dataset for remote
sensing image understanding, In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 16772-16782 (2023)
3. Tseng, G., Zvonkov, I., Purohit, M., Rolnick, D., Kerner, H.: Lightweight, pre[trained transformers for remote sensing timeseries, arXiv:2304.14065 (2023)](http://arxiv.org/abs/2304.14065)
4. Fibaek, C., Camilleri, L., Luyts, A., Dionelis, N., Le Saux, B.: PhilEO Bench: Evaluating Geo-Spatial Foundation Models, in Proc. IGARSS (2024)
5. Le Saux, B., Fibaek, C., Camilleri, L., Luyts, A., Dionelis, N., et al.: The PhilEO
Geospatial Foundation Model Suite, EGU (2024)
6. Chen, S., Long, G., et al.: Foundation models for weather and climate data under[standing: A comprehensive survey, arXiv:2312.03014 (2023)](http://arxiv.org/abs/2312.03014)
7. Xiong, Z., Wang, Y., Zhang, F., et al.: Neural plasticity-inspired foundation model
[for observing the Earth crossing modalities, arXiv:2403.15356 (2024)](http://arxiv.org/abs/2403.15356)
8. Zhu, X.X, Xiong, Z., et al.: On the Foundations of Earth and Climate Foundation
[Models, arXiv:2405.04285 (2024)](http://arxiv.org/abs/2405.04285)
9. Lacoste, A., Lehmann, N., et al.: GEO-Bench: Toward foundation models for Earth
[monitoring, arXiv:2306.03831 (2023)](http://arxiv.org/abs/2306.03831)
10. Xiong, Z., Wang, Y., Zhang, F., and Zhu, X.X.: One for all: Toward unified foun[dation models for Earth vision, arXiv:2401.07527 (2024)](http://arxiv.org/abs/2401.07527)
11. Wang, Y., Braham, N., Xiong, Z., et al.: SSL4EO-S12: A large-scale multimodal,
multitemporal dataset for self-supervised learning in Earth observation, IEEE Geoscience and Remote Sensing Magazine, 11(3):98-106 (2023)
12. Guo, X., Lao, J., Dang, B., et al.: SkySense: A multi-modal remote sensing
foundation model towards universal interpretation for Earth observation imagery,
[arXiv:2312.10115 (2023)](http://arxiv.org/abs/2312.10115)
13. Bountos, N., Ouaknine, A., Rolnick, D.: FoMo-Bench: A multi-modal, multi-scale
and multitask forest monitoring benchmark for remote sensing Foundation Models,
[arXiv:2312.10114 (2023)](http://arxiv.org/abs/2312.10114)
14. Xie, E., et al.: SegFormer: Simple and efficient design for semantic segmentation
with Transformers, in Proc. NeurIPS (2021)
15. Dionelis, N., Pro, F., et al.: Learning from Unlabelled Data with Transformers:
Domain Adaptation for Semantic Segmentation of High Resolution Aerial Images,
in Proc. IGARSS (2024)
16. Reed, C., Gupta, R., Li, S., et al.: Scale-MAE: A scale-aware masked autoencoder
for multiscale geospatial representation learning, in Proc. IEEE/CVF International
Conference on Computer Vision (ICCV) (2023)
17. Manas, O., et al.: Seasonal Contrast: Unsupervised pre-training from uncurated
remote sensing data, in Proc. ICCV (2021)
18. Mall, U., Hariharan, B., Bala, K.: Change-aware sampling and contrastive learning
for satellite images, in Proc. IEEE/CVF CVPR (2023)


