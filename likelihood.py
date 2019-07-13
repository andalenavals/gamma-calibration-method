def chi2(pars, hexp, hsim, binlow=None, binup=None):
    fwhmpars, calpars = pars
    from processor import AddFWHM,  Calibrate,  GetScaleFactor, Scale,  FindLowerNoEmptyUpbin,  FindHigherNoEmptyLowbin
    hsim_fwhm = AddFWHM(hsim, 'hist_sim_fwhm',  fwhmpars)
    hsim_fwhm_ch = Calibrate(hsim_fwhm, 'hist_sim_fwhm_cal', calpars, newbinwidth=1, xlow=0)
    if binlow or binup is not None:
        scalefactor =  GetScaleFactor(hexp, hsim_fwhm_ch, binlow=binlow, binup=binup)
        hsim_fwhm_ch_sc = Scale(hsim_fwhm_ch, 'hist_sim_fwhm_cal_sc',  scalefactor )
        for i in range(binlow, binup + 1):
    else:
        binlow = FindHigherNoEmptyLowbin(hexp, hsim_fwhm_ch)
        binup = FindLowerNoEmptyUpbin(hexp, hsim_fwhm_ch)
        scalefactor =  GetScaleFactor(hexp, hsim_fwhm_ch, binlow=binlow, binup=binup)
        hsim_fwhm_ch_sc = Scale(hsim_fwhm_ch, 'hist_sim_fwhm_cal_sc',  scalefactor )
        
