# Normal and cancer tissues are accurately characterised by intergenic transcription at RNA polymerase 2 binding sites

[![DOI:10.1101/2023.03.24.534112](http://img.shields.io/badge/DOI-bioRxiv/2023.03.24.534112-B31B1B.svg)](https://doi.org/10.1101/2023.03.24.534112)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7740073.svg)](https://doi.org/10.5281/zenodo.7740073) `DOI repro`



This is the code repository for the research paper [Normal and cancer tissues are accurately characterised by intergenic transcription at RNA polymerase 2 binding sites](https://doi.org/10.1101/2023.03.24.534112) by de Langen et al. 

## Table of Contents

- [Background](#background)
- [Code re-use and reproducibility](#code-re-use-and-reproducibility)
- [Dependencies](#dependencies)
- [Data](#data)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Background

Intergenic transcription in normal and cancerous tissue is pervasive and incompletely understood. To investigate this activity at a global level, we constructed an atlas of over 180,000 consensus RNA Polymerase II (RNAP2) bound intergenic regions from more than 900 RNAP2 ChIP-seq experiments across normal and cancer samples. Using unsupervised analysis, we identified 51 RNAP2 consensus clusters, many of which map to specific biotypes and identify tissue-specific regulatory signatures. 

We developed a meta-clustering methodology to integrate our RNAP2 atlas with active transcription across 28,797 RNA-seq samples from TCGA, GTEx and ENCODE, which revealed strong tissue- and disease-specific interconnections between RNAP2 occupancy and transcription. 

We demonstrate that intergenic transcription at RNAP2 bound regions are novel per-cancer and pan-cancer biomarkers showing genomic and clinically relevant characteristics including the ability to differentiate cancer subtypes and are associated with overall survival. Our results demonstrate the effectiveness of coherent data integration to uncover and characterise intergenic transcriptional activity in both normal and cancer tissues. 

## Code re-use and reproducibility
We detail how to reproduce our results and how to re-use some parts of our code in `rerun_all/README.md`.


## Dependencies
We provide singularity environments (on Zenodo) and recipes (./env/ folder). Snakemake is required to re-run all analyses.


## Data
The input data used in this project is publicly available at Zenodo and can be found at the following DOI:

`DOI repro`

The output data produced by this project (RNAP2 occupancy atlas, markers, count tables...) is publicly available at Zenodo and can be found at the following DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7740073.svg)](https://doi.org/10.5281/zenodo.7740073)


## License

This project is licensed under the GPL-3.0 license. See the `LICENSE.md` file in the repository for more details.

## Acknowledgements

This work was supported with ; PhD Fellowship to P.D.L. from the French Ministry of Higher Education and Research (MESR); PhD Fellowship to F.H. from the Provence-Alpes-Côte d’Azur Regional Council (Région SUD); Institut National de la Santé et de la Recherche Médicale (INSERM); The Core Cluster of the Institut Français de Bioinformatique (IFB) (ANR-11-INBS-0013) and the Centre de Calcul Intensif d’Aix-Marseille for granting access to its high performance computing resources. The results shown here are based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga, the GTEx project, the ENCODE Consortium, the ENCODE production laboratories, and independent laboratories who followed the Open Science principles and submitted raw ChIP-seq data into repositories.
