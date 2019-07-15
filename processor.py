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
    import random
    from ROOT import TH1I,  TRandom
    seed = random.randint(0, 500)
    ran = TRandom(seed)
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
    import random
    from ROOT import TH1I,  TRandom
    binwidth = h.GetBinWidth(1);
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

    seed = random.randint(0, 500)
    ran = TRandom(seed)
    for i in range(1, nbins + 1):
        for j in range(1, int(h.GetBinContent(i)) + 1):
            h2.Fill( ran.Uniform(h.GetBinLowEdge(i) , h.GetXaxis().GetBinUpEdge(i) ) )
    return h2
      
def Calibrate(h, name, calpars, newbinwidth=None, xlow=None):
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

def GetNoEmptyLowbin(h):
    for i in range(1, h.GetNbinsX() + 1):
        if h.GetBinContent(i) != 0:
            first = i
            break
    return first

def GetNoEmptyUpbin(h):
    for i in range (h.GetNbinsX(), 0,- 1):
        if h.GetBinContent(i) != 0:
            last = i
            break
    return last

def FindHigherNoEmptyLowbin(h1, h2):
    lbin1 = GetNoEmptyLowbin(h1)
    lbin2 = GetNoEmptyLowbin(h2)
    if (lbin1>lbin2): return lbin1
    else: return lbin2
def FindLowerNoEmptyUpbin(h1, h2):
    ubin1 = GetNoEmptyUpbin(h1)
    ubin2 = GetNoEmptyUpbin(h2)
    if (ubin1>ubin2): return ubin2
    else: return ubin1

def Scale(h, name, scale ):
    h2= h.Clone(name)
    h2.Scale(scale) #new=scale*old
    return h2

def GetScaleFactor(href, hscale, binlow=None, binup=None):
    import numpy as np
    if binlow or binup is not None:
        iref = np.array([href.GetBinContent(i) for i in range(binlow, binup + 1)]).sum()
        sref = np.array([hscale.GetBinContent(i) for i in range(binlow, binup + 1)]).sum()
    else:
        iref = np.array([href.GetBinContent(i) for i in range(1, href.GetNbinsX() + 1)]).sum()
        sref = np.array([hscale.GetBinContent(i) for i in range(1, hscale.GetNbinsX() + 1)]).sum()
    return iref/sref
        
    
    
