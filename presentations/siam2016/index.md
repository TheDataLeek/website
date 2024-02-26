---
layout: post
title: PACE - SIAM 2016
nav-menu: true
show_tile: false
---

Parameterization and Analysis of Viscous Fluid Conduit Edges for Dispersive Hydrodynamics
Zoe Farmer

University: University of Colorado, Boulder

Advisors/Collaborators: Mark Hoefer, Michelle Maiden, Peter Wills

Solitary waves, dispersive shock waves, and other coherent structures can be observed in the viscous fluid conduit system, a deformable, fluid-filled pipe that results from pumping a buoyant, viscous interior fluid into a dense, more viscous exterior fluid.  Conduits are observed experimentally as a vertical column where the interior fluid is dyed black for contrast with the colorless exterior fluid. This work will resolve the problem of analyzing large-scale conduits that display curvature effects using a smoothing regression combined with machine learning techniques.

Initial data is acquired from micro- and macro- scale high-resolution photographs which are pipelined through an edge-detection algorithm, yielding the left and right edges of the conduit averaged to get the mean conduit value. This data can display a variety of behavior, from large-scale horizontal changes to small scale fluctuations. Noise is realized in several ways during this process, ranging from imperfections in the fluid pumping mechanism to the camera focus, and as a result a smoothing method must be applied in order to analyze the data properly. The smoothing method model uses overlapping Gaussian curves with varying heights to remove the noise fluctuations from the dataset and yields an analytic line. Then by using a nondimensionalized curvature value and the residuals from our smoothing process, we can classify the “straightness” of the conduit in question. From this point more novel machine learning classification algorithms can be utilized to determine the data quality of arbitrary points in our classification space.

Parameterization of the Gaussian smoothing process focuses on repeatability and consistency between different datasets by finding the parameters that minimize the standard deviation of the residuals. This method is self-improving and after more data is classified by hand the results will improve. Initial data for model and process testing was acquired from past experiments pending publication.


<iframe src="/presentations/siam2016/PACE.pdf" style="width: 100%; height: 25em;"></iframe>

