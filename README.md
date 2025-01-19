# FALCON (AR-BERT model utilizing Feature TrAnsfer and Multi-Task Learning strategies for extracting spatio-temporal Life Interactioons)
This is the source code for paper
# Abstract
Interactions among notable individuals—whether examined individually, in groups, or as networks—often convey significant messages across cultural, economic, political, scientific, and historical perspectives. By analyzing the times and locations of these interactions, we can observe how dynamics unfold across regions over time. However, relevant studies are often constrained by data scarcity, particularly concerning the availability of specific location and time information. To address this issue, we mine millions of biography pages from Wikipedia, extracting 594,011 interaction records in the form of (Person1, Person2, Time, Location) interaction quadruplets. The key elements of these interactions are often scattered throughout the heterogeneous crowd-sourced text and may be loosely or indirectly associated. We overcome this challenge by designing a model that integrates attention mechanisms, multi-task learning, and feature transfer methods, achieving an F1 score of 86.51%, which outperforms baseline models.  Additionally, we create a manually annotated dataset, WikiInteraction, comprising 4,507 interaction quadruplets. We conduct empirical analyses on the extracted data to showcase its potential. We make our code, the extracted interaction data, and the WikiInteraction dataset publicly available.

# Prerequisites
The code has been successfully tested in the following environment.

*Python 3.9.18
*PyTorch 2.0.1
*numpy 1.22.4
*Pandas 2.1.3
*Transformers 4.44.2

# Training Model
Training Model
Please run following commands for training.

python train_Falcon.py

# Data
We have released a portion of the annotated and extracted interaction data. The remaining data and the complete WikiInteraction dataset will be made available upon acceptance.

# Cite

Please cite our paper if you find this code useful for your research
