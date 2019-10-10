import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('SVA1StyleSheet.mplstyle')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Finding best resolution and energy calibration parameters comparing Geant4 simulations and experiment. Errors in the esmatimated parameters are calculated using MCMC and marginalization of the posterior distribution.')
    
    parser.add_argument('--simfile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI2x2_22Na.dat',
                        help='.dat file of the Geant4 Simulation')
    parser.add_argument('--measurefile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI2x2_22Na.dat',
                        help='.dat file of the experimental data')
    parser.add_argument('--simfiles',
                        default=['/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI2x2_22Na.dat',
                        '/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI2x2_137Cs.dat'],
                        help='list of .dat files of the Geant4 Simulation')
    parser.add_argument('--measurefiles',
                        default=['/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI2x2_22Na.dat',
                        '/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI2x2_137Cs.dat'],
                        help='list of .dat file of the experimental data')
    parser.add_argument('--plotspath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method/plots',
                        help='location of the output of the files')
    parser.add_argument('--outpath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method',
                        help='location of the output of the plots')
    parser.add_argument('--mcmc', default=False, action='store_const', const=True, help='Run mcmc to get posterior of parameters')
    parser.add_argument('--nsig', default=1, type=int, 
                        help='How many sigmas for the marginalized confidence interval')
    parser.add_argument('--nsteps', default=1000, type=int, 
                        help='nsteps of MCMC')
    parser.add_argument('--nwalkers', default=100, type=int, 
                        help='nwalkers of MCMC')
    parser.add_argument('--singlesource', default=False, action='store_const', const=True, help='Use only one source at the time for the fitting. (Useful for debugging)')
    parser.add_argument('--testfit', default=False, action='store_const', const=True, help='Introduce manually the fitting, and compare with the simulation')
    parser.add_argument('--plots', default=False, action='store_const', const=True, help='Do plots during the estimation of the parameters')
    args = parser.parse_args()

    return args

#More proper initial guess to not fall in local minima
def FindInitialGuess(hexp, hsim, binlow=None, binup=None):
    import itertools
    import random
    from ROOT import TRandom
    import numpy as np
    from likelihood import chi2
    #If for any reason the minimum is at a certain limit redefine limits around it
    seed = random.randint(0, 500)
    print("Finding a tentative intial guess. Using seed:", seed  )
    ran = TRandom(seed)
    
    chiaux =  np.inf
    for step in range(10000):
        a = ran.Uniform(0,2)
        b = ran.Uniform(0,2)
        c = ran.Uniform(0,2)
        m = ran.Uniform(0,4)
        d = ran.Uniform(-30,30)
        pars = [a, b, c, m, d]
        if (chi2(pars, hexp, hsim, binlow=binlow, binup=binup) < chiaux):
            chiaux = chi2(pars, hexp, hsim, binlow=binlow, binup=binup)
            print(pars, 'chisq:', chiaux)
    return pars
  
        
def main():
    import numpy as np
    from processor import SaveInTH1, AddFWHM, Calibrate, GetScaleFactor, Scale, FindLowerNoEmptyUpbin,  FindHigherNoEmptyLowbin
    from plotter import PrettyPlot,  plot_samplesdist
    from likelihood import minimizeCHI2,  chi2,  MCMC, minimizeCHI2_list,  chi2_list,  TransfSIM
    from ROOT import TCanvas,  gStyle, kFALSE

    args = parse_args()

    outpath = os.path.expanduser(args.outpath)
    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    except OSError:
        if not os.path.exists(outpath): raise

    plotspath = os.path.expanduser(args.plotspath)
    try:
        if not os.path.exists(plotspath):
            os.makedirs(plotspath)
    except OSError:
        if not os.path.exists(plotspath): raise

    if args.plots:
        filename = os.path.join(plotspath, 'animation.gif')
    else:
        canvas = None
    
    if args.singlesource:
        print("Fitting using only one source",  args.simfile)
        #Save data in arrays
        ch_sim, counts_sim = np.loadtxt(args.simfile, dtype=int, unpack=True)
        ch_exp, counts_exp = np.loadtxt(args.measurefile, dtype=int, unpack=True)
        nbins_exp, xlow_exp, xup_exp = len(ch_exp), min(ch_exp), max(ch_exp)
        nbins_sim, xlow_sim, xup_sim = len(ch_sim), min(ch_sim), max(ch_sim)
        hexp = SaveInTH1(ch_exp,counts_exp,'hist_exp',nbins_exp,xlow_exp,xup_exp)
        hsim = SaveInTH1(ch_sim,counts_sim,'hist_sim',nbins_sim,xlow_sim,xup_sim)
        #i_guess =  FindInitialGuess(hexp, hsim, binlow=binlow, binup=binup)
    
        if args.testfit :
            #fwhmpars  #aE + b*sqrt(E)+c. calpars #mx+d
            #NaI 22Na
            fwhmpars = [-0.00261122,   2.40650623 , -0.00716892]
            calpars = [1.28304727,  -17.21383036]
            #fwhmpars = [2.81372869e-02,  1.45986331e+00,  3.83498352e-04]
            #calpars = [1.37090542,  -76.01722641]
            pars = fwhmpars + calpars
            binlow, binup = 1, 2014
            print('chisq_red:', chi2(pars, hexp, hsim, binlow=binlow, binup=binup)/(binup - binlow) )      
        else:
            #i_guess = [1, 1, 1, 1, 1]
            #i_guess = [0.027, 1.2,  0, 1.3 , -15.6] 
            i_guess = [0, 0, 0, 1, 0] 
            #bnds =  ((None, None), (None, None), (None, None), (None, None), (None, None))

            ##FIRST GUESS USING THE WHOLE EXPERIMENTAL SPECTRUM
            '''
            if args.plots: pars, chisq, imags = minimizeCHI2(i_guess, hexp, hsim, filename=filename)
            else: pars, chisq= minimizeCHI2(i_guess, hexp, hsim)
            print('FIRST GUESS. Chisq_nu:',  chisq/nbins_exp)
            fwhmpars = pars[:3]
            calpars =  pars[3:]
            print('FIRST GUESS. FWHM pars a*E + b*np.sqrt(E) + c:',  fwhmpars)
            print('FIRST GUESS. Calibration pars mx +d:',  calpars)
            '''
  
            #DEPURATING SELECTING PROPER HIGH AND LOW LIMIT
            pars = i_guess; imags = []
            chisqold, chisq =  0, 1.e+10
            binlow, binup = None,  None
            binlow_old, binup_old = 0, 0
            filename = os.path.join(outpath, 'pars_history.txt')
            while(binlow!=binlow_old or binup!=binup_old or chisq<chisqold ):
                pars, chisq, pars_hist = minimizeCHI2(pars, hexp, hsim, binlow=binlow, binup=binup, filename=filename)
                print('Chisq_nu:',  chisq/(binup - binlow))
                fwhmpars = pars[:3]
                calpars =  pars[3:]
                print('FWHM pars a*E + b*np.sqrt(E) + c:',  fwhmpars)
                print('Calibration pars mx +d:',  calpars)
                
                binlow_old = binlow; binup_old = binup; chisqold = chisq
                hsim_fwhm = AddFWHM(hsim, 'hist_sim_fwhm',  fwhmpars)
                hsim_fwhm_ch = Calibrate(hsim_fwhm, 'hist_sim_fwhm_cal', calpars, newbinwidth=1, xlow=0)
                binlow = FindHigherNoEmptyLowbin(hexp, hsim_fwhm_ch)
                binup = FindLowerNoEmptyUpbin(hexp, hsim_fwhm_ch)
                print('Depurated binlow and binup:',  binlow, binup)

            #MCMC (GETTING ERRORS OF ESTIMATES)
            if args.mcmc:
                mflags = [True, True]
                samples, chains =  MCMC(pars, hexp, hsim, binlow=binlow, binup=binup, nwalkers=args.nwalkers,  nsteps=args.nsteps,  mflags=mflags)
                mcmcpath = os.path.join(plotspath, 'mcmc_walkers.png')
                parscontoursparth = os.path.join(plotspath, 'par_contours.png')
                if args.plots: plot_samplesdist(samples, chains, mflags, args.nwalkers, args.nsteps, mcmcpath, parscontoursparth )

        #Animation using pars_history
        if(args.plots):
            fig = plt.figure()
            images = []
            for i, x in enumerta(pars_hist):
                hsimaux = TransfSIM(hsim, x, hexp, binlow=binlow, binup=binup)
                ch_sim = [ hsimaux.GetBinCenter(i) for i in range(binlow, binup + 1)]
                counts_sim =  [ hsimaux.GetBinContent(i) for i in range(binlow, binup + 1)]
                ch_exp = [ hexp.GetBinCenter(i) for i in range(binlow, binup + 1)]
                counts_exp =  [ hexp.GetBinContent(i) for i in range(binlow, binup + 1)]
               
                plt1a, = plt.plot(ch_exp, counts_exp, color='red', marker=None, label='Experiment', alpha=1)
                #plt1a  = plt.scatter(ch_exp, counts_exp, color='red', marker=None, label='Experiment', alpha=1)
                plt1b = plt.xlim( [min(ch_exp),max(ch_exp,)] )
                plt1c = plt.ylim(ymin=0)
                if(i == 0): plt.legend(fontsize=14)

                plt2a,  = plt.plot(ch_sim, counts_sim, color='blue', marker=None, label='Simulation', alpha=1)
                #plt2a = plt.scatter(ch_sim, counts_sim, color='blue', marker=None, label='Simulation', alpha=1)
                plt2b = plt.xlim( [min(ch_sim),max(ch_sim,)] )
                plt2c = plt.ylim(ymin=0)
                if(i == 0): plt.legend(fontsize=14)

                ax = plt.gca()
                fwhmpars =  x[: 3]
                calpars =  x[3:]
                text1 = plt.text(0.45, 1.1,  r'$\chi^{2}_{\nu}$ = %.3f'%( chisq/(binupaux - binlowaux) ), fontsize=12, transform= ax.transAxes)
                text2 = plt.text(0.5, 1.01, r'FWHM(E) = %.2fE %+.2f$\sqrt{E}$ %+.2f'%(fwhmpars[0], fwhmpars[1], fwhmpars[2]), fontsize=12, transform= ax.transAxes)
                text3 = plt.text(0, 1.01, r'Channel(E) = %.2fE %+.2f'%(calpars[0], calpars[1]), fontsize=12, transform= ax.transAxes)
                
                images.append([plt1a, plt2a, text1, text2, text3])
                print(len(images))
            ani = animation.ArtistAnimation(fig, images, interval=200, repeat_delay=10000)
            #ani = animation.ArtistAnimation(fig, images, interval=10000, blit=False, repeat=False)
            print("Printing file: ", filename)
            ani.save(filename, writer='imagemagick', fps=2, dpi=300)
                
            

                
        #Plots with the final estimate
        if(args.plots):
            print(binlow, binup)
            hsim_fwhm = AddFWHM(hsim, 'hist_sim_fwhm',  fwhmpars)
            hsim_fwhm_ch = Calibrate(hsim_fwhm, 'hist_sim_fwhm_cal', calpars, newbinwidth=1, xlow=0)
            scalefactor =  GetScaleFactor(hexp, hsim_fwhm_ch, binlow=binlow, binup=binup)
            print('the scale factor is', scalefactor)
            hsim_fwhm_ch_sc = Scale(hsim_fwhm_ch, 'hist_sim_fwhm_cal_sc',  scalefactor )
            ch_exp = [ hexp.GetBinCenter(i) for i in range(binlow, binup + 1)]
            counts_exp =  [ hexp.GetBinContent(i) for i in range(binlow, binup + 1)]
            ch_sim_fwhm = [ hsim_fwhm.GetBinCenter(i) for i in range(binlow, binup + 1)]
            counts_sim_fwhm =  [hsim_fwhm.GetBinContent(i) for i in range(binlow, binup + 1)]
            ch_sim_fwhm_ch = [ hsim_fwhm_ch.GetBinCenter(i) for i in range(binlow, binup + 1)]
            counts_sim_fwhm_ch =  [hsim_fwhm_ch.GetBinContent(i) for i in range(binlow, binup + 1)]
            ch_sim_fwhm_ch_sc = [ hsim_fwhm_ch_sc.GetBinCenter(i) for i in range(binlow, binup + 1)]
            counts_sim_fwhm_ch_sc =  [hsim_fwhm_ch_sc.GetBinContent(i) for i in range(binlow, binup + 1)]
            #print(hsim_fwhm_ch.GetNbinsX())
            
            #Original Simulation
            plt.clf()
            filepath = os.path.join(plotspath, 'original_sim.png')
            PrettyPlot(ch_sim, counts_sim, color='blue', marker=None, label='Simulation', xlabel='Energy (keV)', ylabel='Counts', alsize=24, legendsize=24)
            print("Printing file: ", filepath)
            plt.savefig(filepath, dpi=200)
            #Original Experiment
            plt.clf()
            filepath = os.path.join(plotspath, 'original_exp.png')
            PrettyPlot(ch_exp, counts_exp, color='red', marker=None, label='Experiment', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=24)
            print("Printing file: ", filepath)
            plt.savefig(filepath, dpi=200)
            #Sim with FWHM
            plt.clf()
            filepath = os.path.join(plotspath, 'sim_fwhm.png')
            PrettyPlot(ch_sim_fwhm, counts_sim_fwhm, color='blue', marker=None, label='Simulation with GEB', xlabel='Energy (keV)', ylabel='Counts', alsize=24, legendsize=18)
            print("Printing file: ", filepath)
            plt.savefig(filepath, dpi=200)
            #Sim with FWHM in channels
            plt.clf()
            filepath = os.path.join(plotspath, 'sim_fwhm_ch.png')
            PrettyPlot(ch_sim_fwhm_ch, counts_sim_fwhm_ch, color='blue', marker=None, label='Simulation with GEB', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=15)
            print("Printing file: ", filepath)
            plt.savefig(filepath, dpi=200)
            #Comparison exp with sim with FWHM and in channels
            plt.clf()
            filepath = os.path.join(plotspath, 'exp_vs_sim_fwhm_ch_sc.png')
            PrettyPlot(ch_exp, counts_exp, color='red', marker=None, label='Experiment', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=15, alpha=.7)
            PrettyPlot(ch_sim_fwhm_ch_sc, counts_sim_fwhm_ch_sc, color='blue', marker=None, label='Simulation with GEB', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=15, alpha=.8)
            plt.title(r'2$\times$2 NaI detector with a $^{137}$Cs source')
            plt.tight_layout()
            print("Printing file: ", filepath)
            plt.savefig(filepath, dpi=200)

    else:
        print("Fitting using", len(args.simfiles) , " sources:",  args.simfiles)
        #Concatenation all the spectra of the same detector for a unique fit
        hexp_list, hsim_list = [], []
        nexps = len(args.simfiles)
        for i in range(nexps):
            ch_sim, counts_sim = np.loadtxt(args.simfiles[i], dtype=int, unpack=True)
            ch_exp, counts_exp = np.loadtxt(args.measurefiles[i], dtype=int, unpack=True)
            hexp_list.append(SaveInTH1(ch_exp, counts_exp, 'hist_exp_%d'%(i), len(ch_exp), min(ch_exp), max(ch_exp)))
            hsim_list.append(SaveInTH1(ch_sim, counts_sim, 'hist_sim_%d'%(i), len(ch_sim), min(ch_sim), max(ch_sim)))
        if args.testfit :
            #fwhmpars  #aE + b*sqrt(E)+c. calpars #mx+d
            #NaI 2x2
            fwhmpars = [2.81372869e-02,  1.45986331e+00,  3.83498352e-04]
            calpars = [1.37090542,  -76.01722641]
            pars = fwhmpars + calpars
            binlow_list, binup_list = [1, 1], [2014, 2048]
            ndof =  (np.array(binup_list) - np.array(binlow_list)).sum()
            print('chisq_red:', chi2_list(pars, hexp_list, hsim_list, binlow_list=binlow_list, binup_list=binup_list)/ndof)      
        else:
            #i_guess = [1, 1, 1, 1, 1]
            #i_guess = [0.027, 1.2,  0, 1.3 , -15.6]
            i_guess = [0, 0, 0, 1, 0] 
            #bnds =  ((None, None), (None, None), (None, None), (None, None), (None, None))

            ##FIRST GUESS USING THE WHOLE EXPERIMENTAL SPECTRUM
            pars, chisq = minimizeCHI2_list(i_guess, hexp_list, hsim_list)
            ndof =  np.array([hexp_list[i].GetNbinsX for i in range(nexps)]).sum()
            print('FIRST GUESS. Chisq_nu:',  chisq/ndof)
            fwhmpars = pars[:3]
            calpars =  pars[3:]
            print('FIRST GUESS. FWHM pars a*E + b*np.sqrt(E) + c:',  fwhmpars)
            print('FIRST GUESS. Calibration pars mx +d:',  calpars)
  
            #DEPURATING SELECTING PROPER HIGH AND LOW LIMIT
            binlow_list, binup_list = [1 for i in range(nexps)], [hexp_list[i].GetNbinsX() for i in range(nexps)]
            binlow_new_list, binup_new_list = [0 for i in range(nexps)], [0 for i in range(nexps)]
            while(binlow_new_list!=binlow_list and binup_new_list!=binup_list):
                print('Entering in depurating mode')
                binlow_list = binlow_new_list; binup_list = binup_new_list
                hsim_fwhm_list = [ AddFWHM(hsim_list[i], 'hist_sim_fwhm_%d'%(i), fwhmpars) for i in range(nexps)]
                hsim_fwhm_ch_list = [Calibrate(hsim_fwhm_list[i], 'hist_sim_fwhm_cal_%d'%(i), calpars, newbinwidth=1, xlow=0) for i in range(nexps)] 
                binlow_new_list = [FindHigherNoEmptyLowbin(hexp_list[i], hsim_fwhm_ch_list[i]) for i in range(nexps)] 
                binup_new_list = [FindLowerNoEmptyUpbin(hexp_list[i], hsim_fwhm_ch_list[i]) for i in range(nexps)] 
                print('Depurated binlow_list and binup_list:',  binlow_new_list, binup_new_list)
                pars, chisq = minimizeCHI2_list(pars, hexp_list, hsim_list, binlow_list=binlow_new_list, binup=binup_new_list)
                ndof =  (np.array(binup_list) - np.array(binlow_list)).sum()
                print('Depurated. Chisq_nu:',  chisq/ndof)
                fwhmpars = pars[:3]
                calpars =  pars[3:]
                print('Depurated. FWHM pars a*E + b*np.sqrt(E) + c:',  fwhmpars)
                print('Depurated. Calibration pars mx +d:',  calpars)
                
            #MCMC (GETTING ERRORS OF ESTIMATES)
            #mflags = [True, True]
            #samples, chains =  MCMC(pars, hexp, hsim, binlow=binlow, binup=binup, nwalkers=args.nwalkers,  nsteps=args.nsteps,  mflags=mflags)
            #mcmcpath = os.path.join(plotspath, 'mcmc_walkers.png')
            #parscontoursparth = os.path.join(plotspath, 'par_contours.png')
            #if args.plots: plot_samplesdist(samples, chains, mflags, args.nwalkers, args.nsteps, mcmcpath, parscontoursparth )
        
        print('Depurated binlow_list and binup_list:', binlow_list, binup_list)
        hsim_fwhm_list = [ AddFWHM(hsim_list[i], 'hist_sim_fwhm_%d'%(i), fwhmpars) for i in range(nexps)]
        hsim_fwhm_ch_list = [Calibrate(hsim_fwhm_list[i], 'hist_sim_fwhm_cal_%d'%(i), calpars, newbinwidth=1, xlow=0) for i in range(nexps)] 
        scalefactor_list =  [GetScaleFactor(hexp_list[i], hsim_fwhm_ch_list[i], binlow=binlow_list, binup=binup) for i in range(nexps)]
        print('the scale factor list is', scalefactor_list)
        hsim_fwhm_ch_sc_list = [Scale(hsim_fwhm_ch_list[i], 'hist_sim_fwhm_cal_sc_%d'%(i),  scalefactor_list[i] ) for i in range(nexps)]
        ch_exp_list = [[ hexp_list[j].GetBinCenter(i) for i in range(binlow_list[j], binup_list[j] + 1)] for j in range(nexps)]
        counts_exp_list =  [ [hexp_list[j].GetBinContent(i) for i in range(binlow_list[j], binup_list[j] + 1)] for j in range(nexps)]
        ch_sim_fwhm_ch_sc_list = [ [hsim_fwhm_ch_sc_list[j].GetBinCenter(i) for i in range (binlow_list[j], binup_list[j] + 1)] for j in range(nexps)]
        counts_sim_fwhm_ch_sc_list =  [ [hsim_fwhm_ch_sc_list[j].GetBinContent(i) for i in range(binlow_list[j], binup_list[j] + 1)] for j in range(nexps)]
   
        if(args.plots):
            names_list = ['NaI2x2_22Na.png', 'NaI2x2_137Cs.png']
            titles_list  = [r'2$\times$2 NaI response function to $^{22}$Na source', r'2$\times$2 NaI response function to $^{137}$Cs source']
            for i in range(nexps):
                plt.clf()
                PrettyPlot(ch_exp_list[i], counts_exp_list[i], color='red', marker=None, label='Experiment', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=15, alpha=.7)
                PrettyPlot(ch_sim_fwhm_ch_sc_list[i], counts_sim_fwhm_ch_sc_list[i], color='blue', marker=None, label='Simulation', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=15, alpha=.8)
                plt.title(titles_list[i])
                plt.tight_layout()
                filepath = os.path.join(plotspath, names_list[i])
                print("Printing file: ", filepath)
                plt.savefig(filepath, dpi=200)




                             
if __name__ == "__main__":
    main()


''' Note Simulation of the 3x3 detector is missing of the 1200 keV
peak do not include in the analysis.
'''
