# Vaccine Failure Modes and Vaccine Escape

Vaccine Failure Mode Model
This repository contains code and figures for a mathematical modelling study that examines how all-or-nothing (AoN) and leaky vaccine protection affect escape mutant selection across a range of disease/population parameters.

**Required Packages**

- solve_ivp from scipy.integrate 
- numpy 
- math
- matplotlib.pyplot
- pandas as pd
- seaborn


**Repository Structure**

- functions.py ---- All model functions, including solvers, metrics, plots

- models/ --------- Contains files with bar plots of of vaccine/mutant impact over various high, medium and low values, along with heatmaps

  -  delayed_vacc_highmedlows -- delayed vaccination scenario
  
  -  delayed_mut_highmedlows --- delayed mutation scenario

  -  undelayed_highmedlows ----- no delay timing scenario
  
Authors:
Cindy (Qin Yi) Yu
Nicole Mideo
Alison Hill
 
