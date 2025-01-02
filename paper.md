---
title: 'ParticlePhaseSpace: A python package for streamlined import, analysis, and export of particle phase space data'
tags:
  - Python
  - accelerator physics
  - topas-mc
  - phase space
authors:
  - name: Brendan Whelan
    orcid: 0000-0002-2326-0927
    affiliation: 1
  - name: Leo Esnault
    orcid: 0000-0002-0795-9957
    affiliation: 2
  
affiliations:
 - name: Image-X Institute, School of Health Sciences, University of Sydney
   index: 1
 - name: Univ. Bordeaux-CNRS-CEA, Centre Lasers Intenses et Applications, UMR 5107, 33405 Talence, France 
   index: 2
date: 22 January 2023
bibliography: paper.bib
---

# Summary

In accelerator particle physics, a description of the positions and directions of an ensemble of particles is defined as phase space (@wiedemann_particle_2015). To describe a given particle at a given time six properties are required, for example, position and momentum: `[x, y, z, px, py, pz]`. To extend this description to arbitrary particles at any arbitrary point in time and to handle particles of different statistical weight the particle species (e.g. electron, X-ray, etc.), time at which its properties were recorded, and statistical weight of the particle must also be included, resulting in a nine dimensional space. Phase space data is commonly both the input and output of particle accelerator simulations. Unfortunately, there is no widely accepted format for phase space data.

# Statement of need

Although the use of phase space data is well established, there is no consistent implementation of phase space data between different programs, as discussed at length in @tessier_proposal_2021. To appreciate why this is an issue, one must understand that in a typical accelerator workflow, it is common to utilize several different programs to simulate different phases of particle transport. For any given simulation task, there are many different simulation programs one can use. An incomplete list is supplied in the Wikipedia article @noauthor_accelerator_2022 and the campa consortium data base @noauthor_acceleration_nodate.  Each of these programs will typically utilize their own unique format for saving and loading phase space data. This means that getting these programs to 'speak' to each other generally requires a substantial amount of work. The openPMD initiative aims to solve this problem by defining a common open-access metadata standard (@openPMDstandard). However, many codes have yet to adopt this standard, and some commercial codes in particular may never do so. The fragmented nature of phase space data formats has in turn led to fragmented analysis frameworks, with different fields and research groups tending to develop their own, often in-house, code for handling phase space data.

`ParticlePhaseSpace` aims to solve these issues by providing well documented, extensible mechanisms for the import and export of data in different formats, as well as a library of methods for visualizing, manipulating, characterizing and analyzing phase space data. There are many examples in the scientific literature where ParticlePhaseSpace would be useful, for example, @st_aubin_integrated_2010, @whelan_novel_2016, and @lesnat_particle_2021. The basic code hierarchy is shown in \autoref{figure 1}.

![Code structure overview.\label{figure 1}](figure_1.png)

Import/Export of different data formats is facilitated through the use of Abstract Base Classes to generate new `DataLoader` and `DataExporter` objects, which ensures consistent data formatting and easy extensibility. The underlying PhaseSpace data is stored in a `pandas` dataframe with configurable and clearly defined required and allowed (optional) quantities.  All optional quantities must have an associated method defining their calculation inside `PhaseSpace.fill`. If this method is not defined and callable the code tests will fail. Similarly, all allowed particles and allowed quantities must be documented, or the code tests will fail. The main `PhaseSpace` object contains various methods for plotting (one and two dimensional histograms, scatter plots, etc.), transforming (translation, rotation, and re-gridding), and filtering (boolean index, time) Phase Space data. `PhaseSpace` objects can also be added or subtracted from each other.  Users can work with a wide range of pre-defined unit sets as well as define new units. The particles which are handled by this code are defined inside `ParticleConfig` which enables simple extension to arbitrary particles. \autoref{figure 2} shows examples of some of the plots which can be generated using `ParticlePhaseSpace` using data from the X-ray collimator described in @whelan_bayesian_2022.

![Examples of plots from ParticlePhaseSpace. A) Multi-particle energy histogram. B) 2D intensity histogram of gamma particles. C) Trace-space in `x` of gamma particles.\label{figure 2}](figure_2.png)

There are some additional open source codes providing similar functionality to this code, including `p2sat` (@lesnat_particle_2021) and `postpic` (@kuschel_postpic_2022). The codes @noauthor_openpmd-viewer_2023 and @mayes_openpmd-beamphysics_2023 have been specifically designed to work with the openPMD standard. `ParticlePhaseSpace` provides an additional tool in this landscape and provides extension mechanisms for loading and exporting arbitrary data formats, a testing framework with continuous integration, multi-particle support in the same Phase Space object, many useful plotting and analysis routines, and automatic code documentation. The major limitation of the code at the time of writing is the ability to handle data larger than memory. Future versions could address this limitation by adding well supported 'chunking' mechanisms or using libraries such as `dask` @DASK to enable distributed operations.

# Acknowledgements

Brendan Whelan acknowledges funding support from the NHMRC, and Julia Johnson (Image-X institute, University of Sydney, Australia) for assistance with figure creation.

# References
