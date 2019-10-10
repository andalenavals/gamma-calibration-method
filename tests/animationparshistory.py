import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('SVA1StyleSheet.mplstyle')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Introduce manually the fitting parameters and compora with sim')
    
    parser.add_argument('--simfile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI2x2_22Na.dat',
                        help='.dat file of the Geant4 Simulation')
    parser.add_argument('--measurefile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI2x2_22Na.dat',
                        help='.dat file of the experimental data')
    parser.add_argument('--parshistory',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/pars_history.txt',
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
    fwhmpars = [parsfile['a'], parsfile['b'], parsfile['c']]
    calpars = [ parsfile['m'], parsfile['d']]
    pars = fwhmpars + calpars
    binlow, binup = parsfile['binlow'], parsfile['binup']
    print('chisq_red:', chi2(pars, hexp, hsim, binlow=binlow, binup=binup)/(binup - binlow) )

    #Animation using pars_history
    if(args.plots):
        fig = plt.figure()
        images = []
        for i, x in enumerate(pars_hist):
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


if __name__ == "__main__":
    main()
