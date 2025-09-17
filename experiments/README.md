# Calculations

This folder contains the code and data to recreate the figures in the paper, as well as to recreate the full results from scratch.

## Recreating the figures

In order to recreate the figures, simply call
```
python3 [path_to_experiment]/plot.py
```
where `path_to_experiment` is one of the directories in `experiments/results`.
The resulting figure(s) will be placed in the same directory as .pdfs.
Do note that the figures appearing in the paper have been subsequently edited for easier readability, mainly in how the legend is presented, and the generated figures will not have these extra touches.

## Recreating the experiments from scratch

Recreating the results using the scripts provided assumes that you are operating in a `bash` environment with the [`slurm`](https://slurm.schedmd.com/overview.html) workload manager available.

To recreate the full results from scratch, start by downloading the relevant parts of the [ACDB](https://github.com/elmjonas/ACDB/tree/master) dataset.
The subdirectories required are
```
Articles/kubecka24_neural_network/Databases # SA-W clusters
Articles/knattrup23_multiacid_multibase/ # clusteromics
```
The datasets required for the QM9 and SA-W extrapolation calculations are included as part of this repository.

Next, ensure that you have the [JKCS](https://jkcs.readthedocs.io/en/latest/) software package installed and configured to your setup, with a version higher than 2.1.
Make sure to update the installation path to `config.sh` at the root of this repository.

First, generate the datasets used in the calculations by calling
```
bash recreate_datasets.job
```
in the relevant directory.
This populates the data directory with the hyperparameter optimisation dataset, as well as the cross-validation chunks at different training set sizes.

You are now ready to perform the machine learning part.
Call
```
bash experiments/scripts/[experiment_script.job]
```
to schedule the jobs using `slurm`.
The first call to the `experiment_script` will train and test a KRR model and perform a hyperparameter search for the $k$-NN models.
After there are finished, call
```
bash experiments/scripts/[experiment_script.job]
```
to perform $k$-NN learning using the acquired hyperparameters.
The experiment scripts also accept a number of options, such as limiting training to only certain representations and training set sizes.
See the scripts for more detail.

Finally, collect the results for plotting using
```
python3 experiments/collect_learning_curve.py [source_dir] [save_dir]
```
where `source_dir` is a directory in `experiments/results` and `save_dir` is a destination.
This will generate a database called `learning_curve.pkl`, containing the error, train, and test times at different train set sizes.

To study the effect of the value of $k$ (Fig. 5), first generate the input files using
```
sbatch experiments/scripts/exp_k_sensitivity.job [source_dir] [representation]
```
where `source_dir` has the results from the machine learning and `repr` is the desired representation.

To recreate the uncertainty estimation analysis, first run the machine learning pipeline for $\Delta$-learning on SA-W clusters.
Then, call
```
sbatch experiments/scripts/exp_uncertainty.job [source_dir] [representation]
```
to create the uncertainty estimation results.