# BatchCP_Classification

This repository contains the code used for the numerical experiments of the following paper:

U. Gazin, R. Heller, E. Roquain, A. Solari "<a href="https://arxiv.org/abs/2411.02239">Powerful batch conformal prediction for classification</a>".

The code is under MIT Licence but please refer to and cite the above paper if you use it for academic purposes.

## Disclaimer on the packages

The folder "Source" contains two python package (), which contains some general conformal prediction codes. They can be found in A. Marandon page: https://github.com/arianemarandon/infoconf.

## Folder: "Python_RealData"

This folder contains some general code to use the batch conformal prediction methodology, with the various procedure decribed in the paper, some applications on the CIFAR and USPS dataset, and the code to obtain the illustrations on real data sets (Section 4.2). The notebook "Python_Batch_GHRS25.ipynb" contains the general methodology and the experiments on the USPS and CIFAR datset. Inside the folder "Illustration", the notebook "Illustration_AISTAS25.ipynb" contains the code to make the illustration in the paper "Powerful batch conformal prediction for classification". The ".npy" are the result of some experiments donne with the codes from "Python_Batch_GHRS25.ipynb".

## Folder: "R_SyntheticData"

This folder contains the code  for the Bivariate Normal Simulations (Section 4.1) simulations, to reproduce the tables in F.4 ("NumericalExperimentsSectionGaussianMultivariateSetting") and in D.4 ("NumericalExperimentsSectionLargeBatches") in R Markdown format. 
