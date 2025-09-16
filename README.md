# JK-kNN
This repository contains the code necessary to run the calculations presented in our paper, Fast and Interpretable Machine Learning Modelling of Atmospheric Molecular Clusters.

## Citations
Preprint of the ExplainReduce paper

> *Seppäläinen, Lauri, Kube\v{c}ka, Jakub, Elm, Jonas, and Puolamäki, Kai (2025)*.
>
> **Fast and Interpretable Machine Learning Modelling of Atmospheric Molecular Clusters**
>
> Arxiv preprint [https://doi.org/10.48550/arXiv.2509.11728](https://doi.org/10.48550/arXiv.2509.11728).

# Requirements
To run the calculations, a working and up-to-date installation of [JKCS](https://jkcs.readthedocs.io/en/latest/) (version 2.1 or higher) is needed. 

For other dependencies (for plotting etc.), we recommend the [uv](https://docs.astral.sh/uv/) package manager.
You can install the dependencies by running
```
uv sync
```
after cloning the repository.
Alternatively, install the requirements via pip with
```
pip3 install -r requirements.txt
```

The datasets used are available at [ACDB](https://github.com/elmjonas/ACDB/tree/master).
Note that the .job bash scripts expect particular file paths for the data; you will likely need to edit these to accomodate your personal setup.

# Recreating calculations and included files 

This repository contains the necessary files to
- recreate the calculations from scratch (i.e., the ACDB datasets)
- produce the final plots as they appear in the paper from aggregated results
Intermediary result files are omitted from the repository due to their size.

Detailed instructions for running the calculations can be found in the [experiments](https://github.com/edahelsinki/JK-kNN/tree/main/experiments) directory.