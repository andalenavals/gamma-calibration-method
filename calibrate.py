import os 
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Alpha beta gamma test solving the fitting problem of system of equatiosn, plotting correlations and final correlation function with bias')
    
    parser.add_argument('--simfile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/sims/Sim_NaI2x2_22Na.dat',
                        help='.dat file of the Geant4 Simulation')
    parser.add_argument('--measurefile',
                        default='/data/publishing/gamma_calibration_method/gamma-calibration-method/data/experiment/2015I(NaI2x2)/22Na_HV806_G200_GF0.5_ST_0.5_150s.dat',
                        help='.dat file of the experimental data')
    parser.add_argument('--outpath', default='/data/publishing/gamma_calibration_method/gamma-calibration-method/plots',
                        help='location of the output of the files')
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

def Rebin(h, name, newbinwidth, xlow):
    ran = TRandom(123)
    binw = h.GetBinWidth(1)
    nbins = h.GetNbinsX()
    xup = h.GetXaxis.GetBinUpEdge(nbins)
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
      
def Calibrate(h, name, ):
TH1D *h2Q= (TH1D*) h->Clone(name);  
  nbins= h2Q->GetNbinsX();
  double new_bins[nbin+1];
  for(int i=0; i <= nbin; i++)
    new_bins[i] = pendiente*h2Q->GetBinLowEdge(i+1)+corte ;
h2Q->SetBins(nbin, new_bins);

def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('SVA1StyleSheet.mplstyle')
    #from ROOT import gROOT,  gSystem, TCanvas
    
    args = parse_args()

    outpath = os.path.expanduser(args.outpath)
    try:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
    except OSError:
        if not os.path.exists(outpath): raise
    
    ch_sim,  counts_sim = np.loadtxt(args.simfile, dtype=int,  unpack=True)
    ch_exp,  counts_exp = np.loadtxt(args.measurefile, dtype=int,  unpack=True)

    #Printing Original data.
    #Simulation
    '''
    plt.clf()
    plt.plot(ch_sim, counts_sim, color='blue', marker=None,  label='Simulation')
    plt.xlim( [min(ch_sim),max(ch_sim)] )
    plt.xlabel(r'Energy (keV)', fontsize=24)
    plt.ylabel(r'Counts', fontsize=24)
    plt.legend(loc='best', fontsize=24)
    plt.tight_layout()
    filepath = os.path.join(outpath, 'original_sim.png')
    print("Printing file: ", filepath)
    plt.savefig(filepath)

    #Experiment
    plt.clf()
    plt.plot(ch_exp, counts_exp, color='red', marker=None,  label='Experiment')
    plt.xlim( [min(ch_exp),max(ch_exp)] )
    plt.xlabel(r'Channel', fontsize=24)
    plt.ylabel(r'Counts', fontsize=24)
    plt.legend(loc='best', fontsize=24)
    plt.tight_layout()
    filepath = os.path.join(outpath, 'original_exp.png')
    print("Printing file: ", filepath)
    plt.savefig(filepath)
    '''
    nbins = len(ch_exp)
    xlow =  min(ch_sim); xup = max(ch_sim)
    hsim = SaveInTH1(ch_sim, counts_sim, 'hist_sim', nbins, xlow, xup)
    pars = [0.027, 1.2,  0] #a,b,c
    hsim = AddFWHM(hsim, 'hist_sim_fwhm',  pars)
    
    ch_sim = [ hsim.GetBinCenter(i) for i in range(1, nbins + 1)]
    counts_sim =  [hsim.GetBinContent(i) for i in range(1, nbins + 1)]
    plt.clf()
    plt.plot(ch_sim, counts_sim, color='blue', marker=None,  label='Simulation with GEB')
    plt.xlim( [min(ch_sim),max(ch_sim)] )
    plt.xlabel(r'Energy (keV)', fontsize=24)
    plt.ylabel(r'Counts', fontsize=24)
    plt.legend(loc='best', fontsize=24)
    plt.tight_layout()
    filepath = os.path.join(outpath, 'simTH1fwhm.png')
    print("Printing file: ", filepath)
    plt.savefig(filepath)
    
    
    '''
    c1 =  TCanvas('c1', '', 800, 600)
    c1.SetBottomMargin( 0.15 )
    c1.SetTopMargin( 0.05 )
    c1.SetLeftMargin( 0.15 )
    c1.SetRightMargin( 0.15 )
    
    hsim.Draw('')
    filepath = os.path.join(outpath, 'simFWHM.png')
    c1.Print(filepath)
    '''
        
        
   
if __name__ == "__main__":
    main()
