# Digital Twin for Mapping Land Use in the Next 100 Years

**Author:** Samuel Huggins  
**Project:** Final Year Computing Project  
**University:** University of East Anglia  
**Supervisor:** Dr Mohsin Raza  
**Academic Year:** 2025/26  

---

## Project Overview

This project develops a prototype geospatial digital twin for mapping and projecting land use and land cover change across East Anglia, focusing on Norfolk and Suffolk.

The system uses satellite imagery, labelled land-cover products, engineered environmental drivers, and machine learning to classify land use patterns and explore forward projection behaviour. The final prototype is not a complete operational digital twin with a real-time feedback loop, but it demonstrates the core components required for a digital-twin-style land-use modelling workflow:

- geospatial data acquisition
- feature engineering
- supervised land-cover classification
- spatial prediction
- change detection
- scenario-style forward projection
- visual inspection through GIS artefacts

The project was developed as part of the UEA Final Year Computing Project portfolio.

---

## Current System Scope

The current implementation focuses on:

- **Region of Interest:** East Anglia, specifically Norfolk and Suffolk
- **Prediction Target:** Land Use / Land Cover classes
- **Satellite Data:** Landsat Collection 2 Level 2 Surface Reflectance
- **Ground-Truth Labels:** ESA WorldCover 2020 and 2021
- **Model Type:** Artificial Neural Network classifier
- **GIS Visualisation:** QGIS and exported raster/PNG map outputs
- **Projection Prototype:** Driver-weighted neighbourhood-based forward projection

The project currently classifies known years and uses the resulting maps to support forward projection experiments. The longer-term “100-year” aim is treated as a research direction rather than a fully validated century-scale forecast.

---

## Land-Cover Classes

The model uses a reduced set of ESA WorldCover classes relevant to the selected region:

| Code | Class |
|---:|---|
| 10 | Tree cover |
| 20 | Shrubland |
| 30 | Grassland |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Bare / sparse vegetation |
| 80 | Water |
| 90 | Wetland |

Some classes are naturally rare in the East Anglia ROI, especially shrubland and wetland, which affects model balance and evaluation.

---

## Main Technologies

- **Python 3.11**
- **Poetry** for dependency and environment management
- **Google Earth Engine** for satellite data processing and export
- **Landsat Collection 2 Level 2 Surface Reflectance**
- **ESA WorldCover** for 2020 and 2021 label data
- **TensorFlow / Keras** for ANN training
- **NumPy, Pandas, Rasterio, Matplotlib** for processing and analysis
- **QGIS** for raster inspection, styling, and report artefacts

---

## Implemented Workflow

The project pipeline is structured around the following stages:

1. **Define the Region of Interest**
   - Norfolk and Suffolk are selected as the study area.
   - The ROI is used consistently for Earth Engine export, model training, prediction, and map generation.

2. **Generate Landsat Annual Composites**
   - Landsat imagery is cloud-masked using QA bands.
   - Annual median composites are generated.
   - Surface reflectance scaling is applied.
   - Spectral bands are prepared for feature extraction.

3. **Add Spectral Indices**
   - NDVI
   - NDWI
   - NDBI
   - BSI
   - NDMI

4. **Add Spatial and Environmental Drivers**
   - Slope
   - Elevation
   - Distance to roads
   - Log-transformed distance to roads
   - Distance to water / coast

5. **Sample Training Data**
   - ESA WorldCover 2020 and 2021 are used as label sources.
   - Stratified sampling is used to improve class representation.
   - Exported CSV files are used for model training.

6. **Train ANN Classifier**
   - A neural network is trained using the exported labelled samples.
   - Results are evaluated using accuracy, macro F1, per-class recall, and confusion matrices.

7. **Predict Land-Cover Maps**
   - Trained models are applied spatially to generate full-ROI predicted maps.
   - Outputs are exported as GeoTIFF and PNG files.

8. **Generate Change Maps**
   - Predicted maps from different years are compared.
   - Change maps are used to inspect spatial transitions and classification behaviour.

9. **Apply Post-Processing**
   - A 3x3 majority filter is used to reduce salt-and-pepper noise.
   - Smoothed outputs are compared against raw predictions.

10. **Prototype Forward Projection**
   - A driver-weighted neighbourhood projection approach is used to generate future land-cover maps.
   - This supports the digital-twin-style objective of exploring plausible spatial change.

---

## Repository Structure

```text
Digital-Twin-for-Mapping-Land-use-in-the-next-100-years/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── results/
│   ├── baseline/
│   ├── with_indices/
│   ├── with_slope/
│   ├── with_slope_roads/
│   ├── with_slope_roads_and_water/
│   ├── projections/
│   └── figures/
│
├── src/
│   └── LULC_digital_twin/
│       ├── data/
│       │   └── worldcover_export.py
│       │
│       ├── landsat/
│       │   └── get_annual_landsat_composite.py
│       │
│       ├── models/
│       │   ├── train_ann.py
│       │   └── predict_map.py
│       │
│       ├── maps/
│       │   ├── change_map.py
│       │   ├── smooth_map.py
│       │   └── project_driver_weighted.py
│       │
│       └── utils/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── docs/
│   ├── diagrams/
│   ├── report_figures/
│   └── project_notes/
│
├── pyproject.toml
├── poetry.lock
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd Digital-Twin-for-Mapping-Land-use-in-the-next-100-years
```

### 2. Install dependencies

```bash
poetry install
```

### 3. Activate the environment

```bash
poetry shell
```

Alternatively, commands can be run directly with:

```bash
poetry run python -m <module_name>
```

### 4. Authenticate Google Earth Engine

The project requires access to Google Earth Engine.

```bash
earthengine authenticate
```

Then initialise Earth Engine in the relevant export scripts.

---

## Main Commands

### Export labelled WorldCover / Landsat samples

```bash
poetry run python -m LULC_digital_twin.data.worldcover_export
```

### Train the ANN classifier

```bash
poetry run python -m LULC_digital_twin.models.train_ann
```

### Generate predicted land-cover maps

```bash
poetry run python -m LULC_digital_twin.models.predict_map
```

### Generate change maps

```bash
poetry run python -m LULC_digital_twin.maps.change_map
```

### Apply majority-filter smoothing

```bash
poetry run python -m LULC_digital_twin.maps.smooth_map
```

### Run driver-weighted forward projection

```bash
poetry run python -m LULC_digital_twin.maps.project_driver_weighted
```

---

## Outputs

The system produces several types of output:

- labelled training CSV files
- trained ANN model files
- classification reports
- confusion matrices
- training accuracy/loss plots
- predicted land-cover GeoTIFFs
- predicted land-cover PNG previews
- smoothed prediction maps
- 2020–2021 change maps
- forward projection maps for future years
- QGIS-styled visual artefacts for the final report and presentation

Example output paths:

```text
results/with_slope_roads_and_water/
results/projections_driver_weighted_neighbourhood/
results/figures/
```

---

## Observed Performance

The best-performing experiments used spectral indices plus spatial drivers such as slope, distance to roads, distance to water, and elevation.

Observed performance reached approximately:

- **Overall accuracy:** around 0.69–0.70
- **Strongest class performance:** water and some major land-cover classes
- **Main confusion issue:** cropland versus bare/sparse vegetation
- **Likely cause of confusion:** annual median Landsat composites over a heavily agricultural region

The model’s performance should be interpreted as a prototype result rather than a production-grade land-cover classifier.

---

## Key Limitations

The current system has several important limitations:

### Limited labelled years

ESA WorldCover provides usable labels for 2020 and 2021 only in this workflow. This restricts temporal validation and long-range forecasting confidence.

### WorldCover as proxy ground truth

ESA WorldCover is itself a classified product, not manually surveyed ground truth. Model evaluation is therefore against a benchmark map rather than direct field labels.

### Agricultural seasonality

East Anglia contains extensive agricultural land. Annual median composites can make cropland appear spectrally similar to bare/sparse vegetation.

### Class imbalance

Some classes, such as shrubland and wetland, are rare within the selected ROI. This makes per-class recall uneven.

### Projection uncertainty

Forward projection is exploratory. It demonstrates a digital-twin-style mechanism but does not claim validated 100-year predictive accuracy.

---

## Future Work

Potential future improvements include:

- adding more historical classified years
- using Sentinel-2 data for higher spatial resolution
- incorporating population, planning, infrastructure, and climate datasets
- testing alternative models such as Random Forest, XGBoost, CNNs, or temporal models
- improving validation with independent land-cover datasets
- implementing scenario-based projection controls
- developing an interactive dashboard or GUI
- introducing a feedback loop where new observations update the model over time

---

## Academic Context

This repository forms part of the supporting material for the UEA Final Year Computing Project portfolio. The final portfolio requires not only a written report, but also supporting artefacts such as source code, data, build instructions, and documentation explaining the structure of the submitted material.

---

## Disclaimer

This project is a research prototype created for academic purposes. The generated maps and projections should not be used for planning, policy, environmental assessment, or commercial decision-making without further validation.