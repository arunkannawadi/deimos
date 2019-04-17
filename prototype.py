## Import packages
import galsim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

## Parameters 
## ----------------------------------------------

gal_snr = 50
dx, dy = 0.0, 0.0 ## sub-pixel offsets, in pixels

galprof_trunc = 0 #4.5
img_size = None #128

## PSF parameters
psf_type = 'moffat'
psf_img_size = 64
psfprof_trunc = 0 #4.5

## Survey parameters
pixel_scale = 0.11

## Metacal parameters
g1, g2 = 0.0, 0.01 ## stimulus

oversampling = 16
x_interpolant = 'exact'
k_interpolant = 'lanczos64'

rng = 12345678

## GSParams
gsparams = {
'xvalue_accuracy': 1e-6,
'kvalue_accuracy': 1e-6,
'maximum_fft_size': 16384,
'minimum_fft_size': 32,
'integration_abserr': 1e-10,
'integration_relerr': 1e-8,
'folding_threshold': 1e-4,
'maxk_threshold': 1e-4,
'stepk_minimum_hlr': 1.0,
}

gsp = galsim.GSParams(**gsparams)

## ----------------------------------------------

def moffat_psf_catalogue(n_psf):
    e1 = np.random.uniform(-0.1,0.1,size=n_psf)
    e2 = np.random.uniform(-0.1,0.1,size=n_psf)
    fwhm = np.random.uniform(0.3,0.5,size=n_psf)
    moffat_beta = np.random.uniform(1.9, 2.6, size=n_psf)

    fieldnames = ['psf_fwhm', 'psf_beta', 'psf_e1', 'psf_e2']
    dat = np.vstack([fwhm,moffat_beta,e1,e2])

    return fieldnames, dat

def generate_galaxy_catalogue(n_gal):
    phi = np.random.uniform(0,1,size=n_gal)*np.pi
    emod = np.random.uniform(0.1,0.9,size=n_gal)
    e1 = emod*np.cos(2*phi)
    e2 = emod*np.sin(2*phi)

    sersicn = np.random.uniform(1,4,size=n_gal)
    hlr = 10.**(np.random.uniform(-1,0.3,size=n_gal))

    fieldnames = ['hlr_circ', 'sersicn', 'gal_e1', 'gal_e2']
    dat = np.vstack([hlr, sersicn, e1, e2])

    return fieldnames, dat

def generate_input_catalogue(n_obj, psf_type='moffat'):
    if psf_type=='moffat':
        psf_fieldnames, psf_dat = moffat_psf_catalogue(n_obj)
    else:
        raise ValueError('psf_type unindentified')
    gal_fieldnames, gal_dat = generate_galaxy_catalogue(n_obj)

    fieldnames = ['id']+psf_fieldnames + gal_fieldnames
    ident = np.linspace(1,n_obj,n_obj)
    input_dat = np.vstack([ident, psf_dat, gal_dat])

    return fieldnames, input_dat.T

def InterpolatedMoments(p,q,img, oversampled_psf_image, gauss_sigma,gauss_centroid=None, gauss_g1=0., gauss_g2=0., x_interpolant='exact', k_interpolant='lanczos64'):
    weight = galsim.Image(np.zeros_like(img.array))
    gauss = galsim.Gaussian(sigma=gauss_sigma*pixel_scale).shear(g1=gauss_g1,g2=gauss_g2)
    if gauss_centroid is None:
        gauss_centroid = img.true_center

    weight = gauss.drawImage(image=weight, scale=pixel_scale, method='no_pixel', use_true_center=True, offset=(gauss_centroid-img.true_center))

    if x_interpolant=='exact':
        x = np.arange(img.xmin-gauss_centroid.x, img.xmax-gauss_centroid.x+0.1,1.)
        y = np.arange(img.ymin-gauss_centroid.y, img.ymax-gauss_centroid.y+0.1,1.)
        X, Y = np.meshgrid(x,y)

        XY = np.vstack([X.flatten(), Y.flatten()])
        A = np.array([[1.+gauss_g1, +gauss_g2], [+gauss_g2, 1.-gauss_g1]])/(1-(gauss_g1**2+gauss_g2**2)) # | det A | = 1
        XYsheared = np.dot(A,XY)
        Xsheared, Ysheared = XYsheared[0,:].reshape(X.shape), XYsheared[1,:].reshape(Y.shape)

        monomial = 1.
        for pp in xrange(p):
            monomial *= Xsheared
        for qq in xrange(q):
            monomial *= Ysheared

        U = -0.5*(Xsheared**2+Ysheared**2)/gauss_sigma**2
        resampled_generalised_weight = galsim.Image(monomial*np.exp(U))/(2.*np.pi*gauss_sigma**2)
    else:
        x = np.arange(img.xmin-img.center.x*0-gauss_centroid.x*1, img.xmax-img.center.x*0-gauss_centroid.x*1+0.1/oversampling, 1./oversampling)
        y = np.arange(img.ymin-img.center.y*0-gauss_centroid.y*1, img.ymax-img.center.y*0-gauss_centroid.y*1+0.1/oversampling, 1./oversampling)
        X, Y = np.meshgrid(x,y)

        assert len(x)==len(y)
        assert X.shape[0]==len(x)
        assert X.shape[1]==len(y)

        oversampled_weight = galsim.Image(np.zeros(( len(x), len(y) )))
        oversampled_weight = gauss.drawImage(image=oversampled_weight, scale=pixel_scale/oversampling, method='no_pixel', use_true_center=True, offset=(gauss_centroid-img.true_center)*(1./oversampling))

        monomial = 1.
        for pp in xrange(p):
            monomial *= X
        for qq in xrange(q):
            monomial *= Y

        if not ((gauss_g1==0.) and (gauss_g2==0.)):
            iPSF = galsim.InterpolatedImage(oversampled_psf_image, x_interpolant=x_interpolant, k_interpolant=k_interpolant, gsparams=gsp)
            invPSF = galsim.Deconvolve(iPSF)
            sheared_psf = iPSF.shear(g1=gauss_g1, g2=gauss_g2)

        generalised_weight_image = galsim.Image(monomial*oversampled_weight.array, scale=pixel_scale/oversampling)
        generalised_weight_func = galsim.InterpolatedImage(generalised_weight_image, x_interpolant=x_interpolant, k_interpolant=k_interpolant, gsparams=gsp)
        sheared_generalised_weight_func = generalised_weight_func.shear(g1=gauss_g1, g2=gauss_g2)

        # Currently not doing the PSF correction. Instead, changing the PSF in the target image
        #if (gauss_g1==0.) and (gauss_g2==0.):
        #    sheared_generalised_weight_psf = sheared_generalised_weight_func
        #else:
        #    sheared_generalised_weight_psf = galsim.Convolve([sheared_generalised_weight_func, sheared_psf, invPSF])
        sheared_generalised_weight_psf = sheared_generalised_weight_func

        resampled_generalised_weight = galsim.Image(np.zeros_like(weight.array))
        resampled_generalised_weight = sheared_generalised_weight_psf.drawImage(image=resampled_generalised_weight, scale=pixel_scale, method='no_pixel', use_true_center=True, offset=(gauss_centroid-img.true_center))

    Q00 = np.sum(weight.array*img.array)
    try:
        Qpq = np.sum(resampled_generalised_weight.array*img.array) #/Q00
        #fig, ax = plt.subplots(1,2)
        #ax[0].imshow(generalised_weight_image.array)
        #ax[1].imshow(resampled_generalised_weight.array)

        return Qpq
    except:
        print len(x), X.shape, x[0], x[-1], X[0,0], X[-1,-1], img.array.shape, resampled_generalised_weight.array.shape
        return -99


def MyBaseMoments(p,q,img,gauss_sigma,gauss_centroid=None, gauss_g1=0., gauss_g2=0.):
    """
    Return flux-normalised generic moments pq
    """
    weight = galsim.Image(np.zeros_like(img.array))
    gauss = galsim.Gaussian(sigma=gauss_sigma*pixel_scale).shear(g1=gauss_g1,g2=gauss_g2)
    if gauss_centroid is None:
        gauss_centroid = img.true_center
    weight = gauss.drawImage(image=weight, scale=pixel_scale, method='no_pixel', use_true_center=True, offset=(gauss_centroid-img.true_center)*(1))
    x = np.linspace(img.xmin-img.center.x*0-gauss_centroid.x*1, img.xmax-img.center.x*0-gauss_centroid.x*1, img.xmax-img.xmin+1)+0.*0.5
    y = np.linspace(img.ymin-img.center.y*0-gauss_centroid.y*1, img.ymax-img.center.y*0-gauss_centroid.y*1, img.ymax-img.ymin+1)+0.*0.5
    X, Y = np.meshgrid(x,y)

    Q00 = np.sum(weight.array*img.array)
    Q10 = gauss_centroid.x + np.sum(X*weight.array*img.array)/Q00
    Q01 = gauss_centroid.y + np.sum(Y*weight.array*img.array)/Q00
    Q20 = np.sum((X**2)*weight.array*img.array)
    Q02 = np.sum((Y**2)*weight.array*img.array)

    monomial = 1.
    for pp in xrange(p):
        monomial *= X
    for qq in xrange(q):
        monomial *= Y
    Qpq = np.sum(monomial*weight.array*img.array) #/Q00

    return Qpq

def MyNativeMoments(img, gauss_sigma, gauss_centroid, gauss_g1=0., gauss_g2=0.):
    weight = galsim.Image(np.zeros_like(img.array))
    gauss = galsim.Gaussian(sigma=gauss_sigma*pixel_scale).shear(g1=gauss_g1,g2=gauss_g2)
    weight = gauss.drawImage(image=weight, scale=pixel_scale, method='no_pixel', use_true_center=True, offset=(gauss_centroid-img.true_center)*(1))
    x = np.linspace(img.xmin-img.center.x*0-gauss_centroid.x*1, img.xmax-img.center.x*0-gauss_centroid.x*1, img.xmax-img.xmin+1)+0.*0.5
    y = np.linspace(img.ymin-img.center.y*0-gauss_centroid.y*1, img.ymax-img.center.y*0-gauss_centroid.y*1, img.ymax-img.ymin+1)+0.*0.5
    X, Y = np.meshgrid(x,y)

    flux = np.sum(weight.array*img.array)
    cent1 = gauss_centroid.x + np.sum(X*weight.array*img.array)/flux
    cent2 = gauss_centroid.y + np.sum(Y*weight.array*img.array)/flux
    num1 = np.sum((X**2-Y**2)*weight.array*img.array)
    num2 = np.sum(2.*(X*Y)*weight.array*img.array)
    den = np.sum((X**2+Y**2)*weight.array*img.array)
    return cent1, cent2, num1/den, num2/den

def workhorse(n_obj):
    awgn = galsim.GaussianNoise(rng=galsim.BaseDeviate(rng))
    fieldnames, input_dat = generate_input_catalogue(n_obj)
    p_array = range(3)
    q_array = range(3)
    output_dat = np.empty((n_obj,2+len(p_array)*len(q_array)*9))
    output_dat[:,0] = np.linspace(1,n_obj,n_obj) ## id
    output_dat[:,1] = np.zeros(n_obj) ## flag

    output_fieldnames = ['ID','Flag']
    for p in p_array:
        for q in q_array:
            for noise_idx in xrange(3):
                output_fieldnames += ['G{0}_{1}_{2}'.format(noise_idx,p,q), 'T{0}_{1}_{2}'.format(noise_idx,p,q), 'M{0}_{1}_{2}'.format(noise_idx,p,q)]

    for idx in xrange(0,n_obj):
        if psf_type=='moffat':
            psf_beta = input_dat[idx][fieldnames.index('psf_beta')]
            psf_fwhm = input_dat[idx][fieldnames.index('psf_fwhm')]
            psf_g1 = input_dat[idx][fieldnames.index('psf_e1')]
            psf_g2 = input_dat[idx][fieldnames.index('psf_e2')]
            psf = galsim.Moffat(beta=psf_beta, fwhm=psf_fwhm, trunc=psfprof_trunc*psf_fwhm).shear(g1=psf_g1,g2=psf_g2)

        col_id = 2

        pix = galsim.Pixel(pixel_scale)
        effective_psf = galsim.Convolve([psf,pix], gsparams=gsp)
        sheared_effective_psf = effective_psf.shear(g1=g1, g2=g2)

        oversampled_psf_image = galsim.Image(psf_img_size*oversampling, psf_img_size*oversampling)
        oversampled_sheared_psf_image = galsim.Image(psf_img_size*oversampling, psf_img_size*oversampling)
        try:
            oversampled_psf_image = effective_psf.drawImage(image=oversampled_psf_image, scale=pixel_scale/oversampling, method='no_pixel')
            oversampled_sheared_psf_image = sheared_effective_psf.drawImage(image=oversampled_sheared_psf_image, scale=pixel_scale/oversampling, method='no_pixel')
        except:
            output_dat[idx][1] = -5
            continue

        gal_n = input_dat[idx][fieldnames.index('sersicn')]
        gal_hlr = input_dat[idx][fieldnames.index('hlr_circ')]
        gal_g1 = input_dat[idx][fieldnames.index('gal_e1')]
        gal_g2 = input_dat[idx][fieldnames.index('gal_e2')]

        try:
            gal = galsim.Sersic(n=gal_n,half_light_radius=gal_hlr, trunc=galprof_trunc*gal_hlr).shear(g1=gal_g1, g2=gal_g2)
            sheared_gal = gal.shear(g1=g1,g2=g2)
            if img_size is None:
                img_noiseless = galsim.Convolve([gal,effective_psf],gsparams=gsp).drawImage(scale=pixel_scale, method='no_pixel',offset=galsim.PositionD(dx,dy))
            else:
                img_noiseless = galsim.Image(img_size, img_size)
                img_noiseless = galsim.Convolve([gal,effective_psf],gsparams=gsp).drawImage(image=img_noiseless, scale=pixel_scale, method='no_pixel', offset=galsim.PositionD(dx,dy))

            sheared_img_noiseless = galsim.Image(img_noiseless.bounds)
            sheared_img_noiseless = galsim.Convolve([sheared_gal,sheared_effective_psf],gsparams=gsp).drawImage(image=sheared_img_noiseless, scale=pixel_scale, method='no_pixel',offset=galsim.PositionD(dx,dy))

            img = img_noiseless.copy()
            img.addNoiseSNR(noise=awgn, snr=gal_snr, preserve_flux=False)
            noise_img = img - img_noiseless

            for pid, p in enumerate(p_array):
                for qid, q in enumerate(q_array):
                    for noise_idx, noise_sgn in enumerate([0,1,-1]):
                        print idx, pid, qid, noise_idx
                        img = img_noiseless + noise_sgn*noise_img
                        sheared_img = sheared_img_noiseless + noise_sgn*noise_img

                        #if noise_sgn==0: ## HACK ALERT: To determine only from noiseless image. Works only when 0 is the first item
                        round_moments = img.FindAdaptiveMom(strict=True, round_moments=True)

                        Gpq = InterpolatedMoments(p,q,img,oversampled_psf_image,round_moments.moments_sigma,round_moments.moments_centroid)
                        Tpq = InterpolatedMoments(p,q,sheared_img,oversampled_sheared_psf_image, round_moments.moments_sigma, round_moments.moments_centroid)
                        Mpq = InterpolatedMoments(p,q,img,oversampled_sheared_psf_image, round_moments.moments_sigma, round_moments.moments_centroid, gauss_g1=g1, gauss_g2=g2)

                        output_dat[idx][0+col_id] = Gpq
                        output_dat[idx][1+col_id] = Tpq
                        output_dat[idx][2+col_id] = Mpq
                        col_id += 3

                        if ((Gpq==-99) or (Tpq==-99) or (Mpq==-99)):
                            output_dat[idx][1] = -99
                        if np.any(np.isnan([Gpq,Tpq,Mpq])):
                            output_dat[idx][1] = -5
        except:
            output_dat[idx][1] = -1

    return output_fieldnames, output_dat

if __name__=='__main__':
    output_fieldnames, output_dat = workhorse(20)
    np.savetxt('prototype_output2.txt', output_dat, header=' '.join(output_fieldnames))
