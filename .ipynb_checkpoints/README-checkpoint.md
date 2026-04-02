# Transboundary Opera

**Author:** Clay Caldwell  
**Contact:** wccaldwe@syr.edu  
**Organization:** Syracuse University  
**Website:** [LinkedIn](https://www.linkedin.com/in/clay-caldwell-9530011a3/)

## Project Overview

- **Problem Statement:** 15 million people rely on the transboundary aquifers spanning the US-Mexico border. While binational regulations have been increasing since the 2006 Transboundary Aquifer Assessmnet Program Act (TAAP) discrepencies in aquifer pumping rates may still be present, impacting water availability for communities on either side of the border.

- **Challenge Statement:** Measuring aquifer pumping across a border of longer than 2000 miles, and with more than 30 distinct aquifers, proves difficult. Historically this type of suveying would be completed with monitoring wells to monitor water levels within the aquifer. On a scale of hundreds of miles, that becomes infeasible.

- **Solution Statement:** Aquifer pumping influences ground surface deformation, and may be identified by changes in surface spectral signature related to evapotranspiration. With this in mind, Interferometric Synthetic Aperture Radar (InSAR) may be used to measure surface displacement across wide spatiotemporal domains. Additionally, the DisALEXI model may be used to invert Landsat imagery to evapotranspiration flux datasets using a two-source energy balance (TSEB).

- **Objective:** Leverage the Alaska Satellite Facility (ASF) OPERA Level-3 Displacement products to extract ground surface deformation signals from 2016 to 2024 across the US-Mexico border. This dataset uses the Sentinel-1 constellation for InSAR to report a time series of ground surface deformation at a spatial resolution of 20 meters and a temporal resolution of as frequent as every 6 days.

## Repository Structure

```
~/local_data
├── code
│   ├── process_data
│   │   └── # Scripts and interactive notebooks used to process the OPERA products into MintPy compatible time series and velocity .h5 files
│   ├── source_data
│   │   └── # Scripts and interactive notebooks used to source the OPERA DISP and CLSC data needed for MintPy
│   └── visualize
│       └── # Scripts and notebooks to create figures, tables, and maps for the LOS and decomposed ground deformation signals
├── raw_data
│   └── # a shapefile used to identify and query data for specific aquifers.
├── analysis_data
│   └── # OPERA data will be stored here upon downloading
├── figures_tables
│   └── # where the figures and tables presenting the results will be stored
├── model_outputs
│   └── # Where the results of the processing will be stored. This will be the MintPy .h5 files
└── src
    └── transboundary_opera
        └── # where the helper functions and classes for the study are located. Functions and classes used to retrieve, process, and assess data for each aquifer
```