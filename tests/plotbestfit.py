import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('SVA1StyleSheet.mplstyle')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Introduce manually the fitting parameters and compora with sim')
    
    parser.add_argument('--simfile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI2x2_137Cs.dat',
                        help='.dat file of the Geant4 Simulation')
    parser.add_argument('--measurefile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI2x2_137Cs.dat',
                        help='.dat file of the experimental data')
    parser.add_argument('--parsfile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/finalfits/singlesource/finalparsNaI2x2_22Na_bc.yaml',
                        help='.yaml with the fitting data')
    parser.add_argument('--plotspath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method/plots',
                        help='location of the output of the files')
    parser.add_argument('--plots', default=True, action='store_const', const=True, help='Do plots during the estimation of the parameters')
    args = parser.parse_args()

    return args

def main():
    import sys; sys.path.append(".")
    import numpy as np
    from processor import SaveInTH1, AddFWHM, Calibrate, GetScaleFactor, Scale, FindLowerNoEmptyUpbin,  FindHigherNoEmptyLowbin
    from plotter import PrettyPlot,  plot_samplesdist
    from likelihood import minimizeCHI2,  chi2,  MCMC, minimizeCHI2_list,  chi2_list,  TransfSIM
    from ROOT import TCanvas,  gStyle, kFALSE
    import yaml

    args = parse_args()

    plotspath = os.path.expanduser(args.plotspath)
    try:
        if not os.path.exists(plotspath):
            os.makedirs(plotspath)
    except OSError:
        if not os.path.exists(plotspath): raise

        
    

    #Save data in arrays
    ch_sim, counts_sim = np.loadtxt(args.simfile, dtype=int, unpack=True)
    ch_exp, counts_exp = np.loadtxt(args.measurefile, dtype=int, unpack=True)
    nbins_exp, xlow_exp, xup_exp = len(ch_exp), min(ch_exp), max(ch_exp)
    nbins_sim, xlow_sim, xup_sim = len(ch_sim), min(ch_sim), max(ch_sim)
    hexp = SaveInTH1(ch_exp,counts_exp,'hist_exp',nbins_exp,xlow_exp,xup_exp)
    hsim = SaveInTH1(ch_sim,counts_sim,'hist_sim',nbins_sim,xlow_sim,xup_sim)
        
    #fwhmpars  aE + b*sqrt(E)+c.
    #calpars  mx+d
    #NaI 22Na
    stream = open(args.parsfile, 'r')
    parsfile = yaml.safe_load(stream.read())
    
    fwhmpars = []; mflags = []
    try: fwhmpars.append(parsfile['a']); mflags.append(True)
    except: print('Parameter a does not exist'); mflags.append(False)
    try: fwhmpars.append(parsfile['b']); mflags.append(True)
    except: print('Parameter b does not exist'); mflags.append(False)
    try: fwhmpars.append(parsfile['c']); mflags.append(True)
    except: print('Parameter c does not exist'); mflags.append(False)
    calpars = [ parsfile['m'], parsfile['d']]
    pars = fwhmpars + calpars
    binlow, binup = parsfile['binlow'], parsfile['binup']
    print('chisq_red:', chi2(pars, hexp, hsim, mflags=mflags, binlow=binlow, binup=binup)/(binup - binlow) )

     #Plots with the final estimate
    if(args.plots):
        print(binlow, binup)
        hsim_fwhm = AddFWHM(hsim, 'hist_sim_fwhm',  fwhmpars, mflags=mflags)
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
        filename = os.path.join(plotspath, 'original_sim.png')
        PrettyPlot(ch_sim, counts_sim, color='blue', marker=None, label='Simulation', xlabel='Energy (keV)', ylabel='Counts', alsize=24, legendsize=24)
        print("Printing file: ", filename)
        plt.savefig(filename, dpi=200)
        #Original Experiment
        plt.clf()
        filename = os.path.join(plotspath, 'original_exp.png')
        PrettyPlot(ch_exp, counts_exp, color='red', marker=None, label='Experiment', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=24)
        print("Printing file: ", filename)
        plt.savefig(filename, dpi=200)
        #Sim with FWHM
        plt.clf()
        filename = os.path.join(plotspath, 'sim_fwhm.png')
        PrettyPlot(ch_sim_fwhm, counts_sim_fwhm, color='blue', marker=None, label='Simulation with GEB', xlabel='Energy (keV)', ylabel='Counts', alsize=24, legendsize=18)
        print("Printing file: ", filename)
        plt.savefig(filename, dpi=200)
        #Sim with FWHM in channels
        plt.clf()
        filename = os.path.join(plotspath, 'sim_fwhm_ch.png')
        PrettyPlot(ch_sim_fwhm_ch, counts_sim_fwhm_ch, color='blue', marker=None, label='Simulation with GEB', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=15)
        print("Printing file: ", filename)
        plt.savefig(filename, dpi=200)
        #Comparison exp with sim with FWHM and in channels
        plt.clf()
        filename = os.path.join(plotspath, 'exp_vs_sim_fwhm_ch_sc.png')
        PrettyPlot(ch_exp, counts_exp, color='red', marker=None, label='Experiment', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=15, alpha=.7)
        PrettyPlot(ch_sim_fwhm_ch_sc, counts_sim_fwhm_ch_sc, color='blue', marker=None, label='Simulation with GEB', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=15, alpha=.8)
        plt.title(r'2$\times$2 NaI detector with a $^{137}$Cs source')
        plt.tight_layout()
        print("Printing file: ", filename)
        plt.savefig(filename, dpi=200)

if __name__ == "__main__":
    main()
