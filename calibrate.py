import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('SVA1StyleSheet.mplstyle')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Alpha beta gamma test solving the fitting problem of system of equatiosn, plotting correlations and final correlation function with bias')
    
    parser.add_argument('--simfile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI3x3_22Na.dat',
                        help='.dat file of the Geant4 Simulation')
    parser.add_argument('--measurefile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/Exp_NaI3x3_22Na.dat',
                        help='.dat file of the experimental data')
    parser.add_argument('--outpath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method/plots',
                        help='location of the output of the files')
    parser.add_argument('--plots', default=False, action='store_const', const=True, help='Do plots during the estimation of the parameters')
    args = parser.parse_args()

    return args

def SaveInTH1(ch_sim, counts_sim, name, nbins, xlow, xup):
    from ROOT import TH1I
    h = TH1I(name, '', nbins, xlow, xup)
    for i in range(len(ch_sim)):
        h.SetBinContent(h.FindBin(ch_sim[i]), counts_sim[i])
    return h

def FWHM(E, modelpars):
    import numpy as np
    a, b, c = modelpars
    return a*E + b*np.sqrt(E) + c 
def AddFWHM(h, name, modelpars):
    from ROOT import TH1I,  TRandom
    ran = TRandom(123)
    nbins = h.GetNbinsX()
    hout = TH1I(name,'', nbins, h.GetBinLowEdge(1), h.GetXaxis().GetBinUpEdge(nbins) )
    for i in range(1, nbins + 1):
        for j in range(1, int(h.GetBinContent(i))+ 1):
            hout.Fill(ran.Gaus(h.GetBinCenter(i), FWHM(h.GetBinCenter(i), modelpars)/2.35  ))
    return hout

def CalFunction(x, calpars):
    slope, intercept = calpars
    return slope*x +  intercept
def Rebin(h, name, newbinwidth, xlow):
    from ROOT import TH1I,  TRandom
    ran = TRandom(123)
    binw = h.GetBinWidth(1)
    nbins = h.GetNbinsX()
    xup = h.GetXaxis().GetBinUpEdge(nbins)
    L = xup - xlow
    if( L%newbinwidth == 0 ):
      newnbin= int(L/newbinwidth)
      h2= TH1I(name, "" , newnbin , xlow , xup  ) 
    if( L%newbinwidth  >0 ):
      newnbin= int(L/newbinwidth)+1 ;
      newxup=xup+newbinwidth;
      h2= TH1I(name , "" , newnbin , xlow , newxup )
    for i in range(1, nbins + 1):
        for j in range(1, int(h.GetBinContent(i)) + 1):
            h2.Fill( ran.Uniform(h.GetBinLowEdge(i) , h.GetXaxis().GetBinUpEdge(i) ) );
    return h2
      
def Calibrate(h, name, calpars, newbinwidth=None, xlow=None ):
    import numpy as np
    #We first create and histogram with no uniform binning 
    h2= h.Clone(name)  
    nbins= h2.GetNbinsX()
    new_bins = np.array([ CalFunction( h2.GetBinLowEdge(i+1), calpars) for i in range (nbins + 1)])
    h2.SetBins(nbins, new_bins)
    #Then we rebin in a uniform bin histogram
    if newbinwidth is not None:
        name = '%s_rebin'%(name)  
        h2 = Rebin(h2, name,  newbinwidth, xlow)
    return h2

def PrettyPlot(x, y, color='black', marker=None, label=None, xlabel='x-axis', ylabel='y-axis', alsize=24, legendsize=24, filepath='outplot.png'):
    plt.plot(x, y, color=color, marker=marker,  label=label)
    plt.xlim( [min(x),max(x)] )
    plt.ylim(ymin=0)
    plt.xlabel(xlabel, fontsize=alsize)
    plt.ylabel(ylabel, fontsize=alsize)
    plt.legend(loc='best', fontsize=legendsize)
    plt.tight_layout()
    print("Printing file: ", filepath)
    plt.savefig(filepath)
    
def main():
    import numpy as np
    args = parse_args()

    outpath = os.path.expanduser(args.outpath)
    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    except OSError:
        if not os.path.exists(outpath): raise

    #Save data in arrays
    ch_sim,  counts_sim = np.loadtxt(args.simfile, dtype=int,  unpack=True)
    ch_exp,  counts_exp = np.loadtxt(args.measurefile, dtype=int,  unpack=True)

    nbins = len(ch_exp)
    xlow =  min(ch_sim); xup = max(ch_sim)
    hsim = SaveInTH1(ch_sim, counts_sim, 'hist_sim', nbins, xlow, xup)
    pars = [0.027, 1.2,  0] #aE + b*sqrt(E)+c
    hsim_fwhm = AddFWHM(hsim, 'hist_sim_fwhm',  pars)
    ch_sim_fwhm = [ hsim_fwhm.GetBinCenter(i) for i in range(1, hsim_fwhm.GetNbinsX() + 1)]
    counts_sim_fwhm =  [hsim_fwhm.GetBinContent(i) for i in range(1, hsim_fwhm.GetNbinsX() + 1)]
    calpars =  [1.3 , -15.6  ] #mx+d
    hsim_fwhm_ch = Calibrate(hsim_fwhm, 'hist_sim_fwhm_cal', calpars, newbinwidth=1, xlow=0)
    ch_sim_fwhm_ch = [ hsim_fwhm_ch.GetBinCenter(i) for i in range(1, hsim_fwhm_ch.GetNbinsX() + 1)]
    counts_sim_fwhm_ch =  [hsim_fwhm_ch.GetBinContent(i) for i in range(1, hsim_fwhm_ch.GetNbinsX() + 1)]
    print(hsim_fwhm_ch.GetNbinsX())

    

    if(args.plots):
        #Original Simulation
        plt.clf()
        filepath = os.path.join(outpath, 'original_sim.png')
        PrettyPlot(ch_sim, counts_sim, color='blue', marker=None, label='Simulation', xlabel='Energy (keV)', ylabel='Counts', alsize=24, legendsize=24, filepath=filepath)
        #Original Experiment
        plt.clf()
        filepath = os.path.join(outpath, 'original_exp.png')
        PrettyPlot(ch_exp, counts_exp, color='red', marker=None, label='Experiment', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=24, filepath=filepath)
        #Sim with FWHM
        plt.clf()
        filepath = os.path.join(outpath, 'sim_fwhm.png')
        PrettyPlot(ch_sim_fwhm, counts_sim_fwhm, color='blue', marker=None, label='Simulation with GEB', xlabel='Energy (keV)', ylabel='Counts', alsize=24, legendsize=18, filepath=filepath)
        #Sim with FWHM in channels
        plt.clf()
        filepath = os.path.join(outpath, 'sim_fwhm_ch.png')
        PrettyPlot(ch_sim_fwhm_ch, counts_sim_fwhm_ch, color='blue', marker=None, label='Simulation with GEB', xlabel='Channel', ylabel='Counts', alsize=24, legendsize=18, filepath=filepath)

    
    
    
    
    
  
    
 
        
        
   
if __name__ == "__main__":
    main()
