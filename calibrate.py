import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('SVA1StyleSheet.mplstyle')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Finding best resolution and energy calibration parameters comparing Geant4 simulations and experiment. Concatinating response function to different sources of the same experimental sutup')
    
    parser.add_argument('--simfiles',
                        default=['/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI2x2_22Na.dat',
                        '/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI2x2_137Cs.dat'],
                        help='list of .dat files of the Geant4 Simulation')
    parser.add_argument('--measurefiles',
                        default=['/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI2x2_22Na.dat',
                        '/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI2x2_137Cs.dat'],
                        help='list of .dat file of the experimental data')
    parser.add_argument('--initial_guess_file',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/finalfits/singlesource/finalNaI2x2_22Na_bc.yaml',
                        help='.yaml with the initial guess')
    parser.add_argument('--filename',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/finalparsNaI2x2_Na.yaml',
                        help='.yaml with the final minimization')
    parser.add_argument('--outpath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data',
                        help='location of the output of the plots')
    parser.add_argument('--history', default=False, action='store_const', const=True, help='Save the history of the parameters while minimizing')
    parser.add_argument('--verbose', default=False, action='store_const', const=True, help='Print parameters during the process of minimization')
    args = parser.parse_args()

    return args
        
def main():
    import numpy as np
    import yaml
    from processor import SaveInTH1,FindNewLimits,  GetNoEmptyLowbin,  GetNoEmptyUpbin
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

    print("Fitting using", len(args.simfiles) , " sources:",  args.simfiles)
    #Concatenation all the spectra of the same detector for a unique fit
    hexp_list, hsim_list = [], []
    nexps = len(args.simfiles)
    for i in range(nexps):
        ch_sim, counts_sim = np.loadtxt(args.simfiles[i], dtype=int, unpack=True)
        ch_exp, counts_exp = np.loadtxt(args.measurefiles[i], dtype=int, unpack=True)
        hexp_list.append(SaveInTH1(ch_exp, counts_exp, 'hist_exp_%d'%(i), len(ch_exp), min(ch_exp), max(ch_exp)))
        hsim_list.append(SaveInTH1(ch_sim, counts_sim, 'hist_sim_%d'%(i), len(ch_sim), min(ch_sim), max(ch_sim)))

    stream = open(args.initial_guess_file, 'r')
    parsfile = yaml.safe_load(stream.read())
    fwhmpars = []; mflags = []
    try: fwhmpars.append(parsfile['a']); mflags.append(True)
    except: print('Parameter a does not exist'); mflags.append(False)
    try: fwhmpars.append(parsfile['b']); mflags.append(True)
    except: print('Parameter b does not exist'); mflags.append(False)
    try: fwhmpars.append(parsfile['c']); mflags.append(True)
    except: print('Parameter c does not exist'); mflags.append(False)
    calpars = [ parsfile['m'], parsfile['d']]
    i_guess =  fwhmpars + calpars
    binlow =  parsfile['binlow']
    binup =  parsfile['binup']
    print('Initial parameters file:',  parsfile)
    print('Initial parameters:',  i_guess)
    
    binlow_list = []; binup_list = []
    for hexp in hexp_list:
        print('Initial low bin divergence',  binlow - GetNoEmptyLowbin(hexp))
        print('Initial up bin divergence',  binup - GetNoEmptyUpbin(hexp))
        binlow_list.append(GetNoEmptyLowbin(hexp))
        binup_list.append(GetNoEmptyUpbin(hexp))
    print('Limit bins used:', binlow_list, binup_list)
    chisq =  chi2_list(i_guess, hexp_list, hsim_list, mflags=mflags, binlow_list=binlow_list, binup_list=binup_list)
    ndof =  (np.array(binup_list) - np.array(binlow_list)).sum()
    print('Initial Chisq_nu:',  chisq/ndof)


    #START THE ITERATION
    pars = i_guess; 
    chisqold, chisq =  np.inf,  0
    n = 0
    if args.history:
        historyfile = os.path.join(outpath, 'pars_history.txt')
        pars_hist =  []
    else:
        historyfile =  None
        pars_hist =  None

    while( chisq<chisqold ):
        print('Recursive iteration number',  n)
        chisqold = chisq
        pars, chisq = minimizeCHI2_list(pars, hexp_list, hsim_list,
                                        mflags=mflags, binlow_list=binlow_list, binup_list=binup_list,
                                        pars_hist=pars_hist, verbose=args.verbose)
       
        calpars =  pars[-2:]
        fwhmpars =  pars[:len(pars)-len(calpars) ]
        print('FWHM pars a*E + b*np.sqrt(E) + c:',  fwhmpars)
        print('Calibration pars mx +d:',  calpars)
        ndof =  (np.array(binup_list) - np.array(binlow_list)).sum()
        print('Chisq_nu:',  chisq/ndof)

        binlow_list = [FindNewLimits(hsim_list[i], hexp_list[i], pars, mflags=mflags)[0] for i in range(nexps)] 
        binup_list = [FindNewLimits(hsim_list[i], hexp_list[i], pars, mflags=mflags)[1]  for i in range(nexps)]
        
        n += 1

                

    data = {}
    af, bf, cf = mflags
    if (af and bf and cf):
        data['a'] = float(pars[0]); data['b'] = float(pars[1]); data['c'] = float(pars[2])
    if (bf and cf):
        data['b'] = float(pars[0]); data['c'] = float(pars[1])
    data['m'] = float(calpars[0]); data['d'] = float(calpars[1])
    data['binlow'] = int(binlow)
    data['binup'] =  int(binup)
    data['chisq'] = float(chisq/(binup - binlow))
    print(data)
    with open(args.filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    print('printing',  args.filename)

    n= [['a','b','c'][i]*f for i,f in enumerate(mflags)]
    header='%s %s %s m d'%(n[0],n[1],n[2])
    
    if args.history:
        np.savetxt(historyfile, pars_hist, header=header,  fmt='%1.4e')
        print(historyfile, 'printed')



                             
if __name__ == "__main__":
    main()


''' Note Simulation of the 3x3 detector is missing of the 1200 keV
peak do not include in the analysis.
'''
