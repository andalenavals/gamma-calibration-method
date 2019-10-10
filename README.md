# Gamma scintillator calibratrion method based on simulations

  Finding best resolution and energy calibration parameters comparing
  Geant4 simulations and experiment. Errors in the estimated
  parameters are calculated using MCMC and marginalization of the
  posterior distribution.

## In the folder data/ are the Geant4 simulations and experimental
   data of the response function of several detector to several
   sources

## processor.py set of functions that allows for the Gausian Energy
   Broadening and Energy calibration

## plotter.py

   Set of function to plot comparison of spectra and contours plots of
   estimated parameters

## likelihood.py

   set of functions defining chisq and likelihood functions.

## calibratesinglesource.py  

   Main code performs the whole analysis using only one response
   function to a calibration source. Recive one response funtions
   simulated and experimental and return the bestfit parameters

   1) --simfile .dat file of the Geant4 simulation used only if --singlesource

   2) --measurefile .dat file of the experimental data used only if --singlesource

   3) --initial_guess_file Initial guess file of the values, ideally
   this should come from a grid long iteration using /tests/gridinitialguess.py

   4) --verbose print parameter each step during minimization

   5) --history save the history of parameters while minimizing in a
   external txt file for evaluation outsi or creation of animation
   
## mcmcsinglesource.py and mcmc.py

   Run mcmc to build posterior (get errors of the estimated
   parameters) you must define:

   1) --nsig The errors sigma confidence of the posterior

   2) --nsteps number of steps each walker gives

   3) --nwalker number of walkers

   4) --initial_guess_file initial guess of the minimum values,
   conviniently coming from scipy first minimization using
   calibratesinglesource.py

## calibrate.py

   Main code that performs the whole analysis using several sources
   simultaneously. It is assumed that experimentally all the
   measurementes were done using the same setup, particularly same
   voltajes and amplifier gain. It recieves more than one response
   function and the same amount of sims, and return best fits.

   1) --simfiles list of .dat files of Geant4 simulations

   2) --measurefiles list of .dat files of experimental data

## In the folder tests there are some test of the fittings:

   1) animationparshistory.py: produce animation using the history of
   th parameter during the minimization

   2) gridinitialguess.py: get the initial guess of the parameters
   evaluating uniformly in random places in the 5 dimensional hypercube.

   3) plotbestfit.py: plot response function of the experiment, sim,
   sim with GEB, sim calibrated and comparison, to compare visually
   how so good a particular set of parameters is.

## HOW TO RUN

   A. First look for a initial guess of the parameters evaluating
   randomlly in a grid. $ python tests/gridinitialguess.py

   B. Once we get and reasonable initial_guess use it an minimize
   chi2. $ python calibratesinglesource.py

   C. Once we get the minimization for a single source, the minization
   using all the sources simultaneously must be close of the pars
   estimated before. $ python calibrate.py

   D. Finally find the error of the parameters. $ python mcmc.py

## PAPER DRAFT

   https://es.overleaf.com/project/5d9d980dd6fb0c0001c1f7d4