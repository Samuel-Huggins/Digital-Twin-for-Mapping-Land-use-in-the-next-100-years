# Digital Twin for Mapping Land Use in the Next 100 Years

**Author:** Samuel Huggins  
**Project:** Final Year Computing Project  
**University:** University of East Anglia  
**Date:** 2025  

---

## Project Overview

This project aims to develop a **Digital Twin** capable of predicting land use for a selected region over the next 100 years.  
It leverages:

- **Google Earth Engine (GEE)** for remote sensing and satellite imagery (Landsat)  
- **Artificial Neural Networks (ANN)** for predictive modelling  
- **Cuckoo Search (CS) algorithm** for hyperparameter optimization  

---

## Repository Structure
Digital-Twin-for-Mapping-Land-use-in-the-next-100-years/
│
├─ data/ # Raw and processed datasets (not tracked in Git)
├─ preprocessing/ # Scripts for downloading and processing Landsat / GEE data
├─ model/ # ANN-CS model scripts and checkpoints
├─ simulation/ # Code for 100-year land use scenario simulation
├─ results/ # Output maps, figures, and evaluation results
├─ notebooks/ # Jupyter notebooks for experimentation
├─ docs/ # Project notes, diagrams, and literature
├─ venv/ # Python virtual environment (not tracked in Git)
├─ .gitignore # Git ignore file
└─ README.md # Project overview and setup instructions
