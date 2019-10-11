import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('SVA1StyleSheet.mplstyle')
from matplotlib import animation

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Introduce manually the fitting parameters and compora with sim')
    
    parser.add_argument('--simfile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI2x2_22Na.dat',
                        help='.dat file of the Geant4 Simulation')
    parser.add_argument('--measurefile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI2x2_22Na.dat',
                        help='.dat file of the experimental data')
    parser.add_argument('--mflags',
                        default=[False, True, True],
                        help='Flags to select parameter of a FWH model')
    parser.add_argument('--parshistory',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/pars_history.txt',
                        help='.yaml with the fitting data')
    parser.add_argument('--plotspath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method/plots',
                        help='location of the output of the files')
    args = parser.parse_args()

    return args

def main():
    import sys; sys.path.append(".")
    import numpy as np
    from processor import SaveInTH1, GetNoEmptyLowbin, GetNoEmptyUpbin
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

    af, bf, cf = args.mflags

    #Save data in arrays
    ch_sim, counts_sim = np.loadtxt(args.simfile, dtype=int, unpack=True)
    ch_exp, counts_exp = np.loadtxt(args.measurefile, dtype=int, unpack=True)
    nbins_exp, xlow_exp, xup_exp = len(ch_exp), min(ch_exp), max(ch_exp)
    nbins_sim, xlow_sim, xup_sim = len(ch_sim), min(ch_sim), max(ch_sim)
    hexp = SaveInTH1(ch_exp,counts_exp,'hist_exp',nbins_exp,xlow_exp,xup_exp)
    hsim = SaveInTH1(ch_sim,counts_sim,'hist_sim',nbins_sim,xlow_sim,xup_sim)
        
  
    history = np.loadtxt(args.parshistory)
    #history = history[: 2]
    #print('cutting tests')
    #print(history)
    binlow = GetNoEmptyLowbin(hexp)
    binup = GetNoEmptyUpbin(hexp)
    
    #Animation using pars_history
    fig = plt.figure()
    images = []
    for i, x in enumerate(history):
        hsimaux = TransfSIM(hsim, x, hexp, mflags=args.mflags,  binlow=binlow, binup=binup)
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
        calpars =  x[-2:]
        fwhmpars =  x[:len(x)-len(calpars) ]
        
        chisq = chi2(x, hexp, hsimaux, mflags=args.mflags,  binlow=binlow, binup=binup)/(binup - binlow)
        print('cal_pars', calpars)
        print('fwhmpars', fwhmpars)
        print('chi_nu:', chisq)
        text1 = plt.text(0.45, 1.1,  r'$\chi^{2}_{\nu}$ = %.3f'%( chisq), fontsize=12, transform= ax.transAxes)
        if(af and bf and cf): text2 = plt.text(0.5, 1.01, r'FWHM(E) = %.2fE %+.2f$\sqrt{E}$ %+.2f'%(fwhmpars[0], fwhmpars[1], fwhmpars[2]), fontsize=12, transform= ax.transAxes)
        if(not af and bf and cf): text2 = plt.text(0.5, 1.01, r'FWHM(E) = %+.2f$\sqrt{E}$ %+.2f'%(fwhmpars[0], fwhmpars[1]), fontsize=12, transform= ax.transAxes)
        text3 = plt.text(0, 1.01, r'Channel(E) = %.2fE %+.2f'%(calpars[0], calpars[1]), fontsize=12, transform= ax.transAxes)
                
        images.append([plt1a, plt2a, text1, text2, text3])
        print(len(images))
    ani = animation.ArtistAnimation(fig, images, interval=200, repeat_delay=10000)
    #ani = animation.ArtistAnimation(fig, images, interval=10000, blit=False, repeat=False)
    filename = os.path.join(plotspath, 'animation.gif')
    print("Printing file: ", filename)
    ani.save(filename, writer='imagemagick', fps=2, dpi=300)


if __name__ == "__main__":
    main()
