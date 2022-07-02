import mne
import eelbrain as eel

# from . import config
from .config import *
from .predictor import *

first_group_subs = ['R2522', 'R2566', 'R2568', 'R2571', 'R2572', 'R2574', 
                    'R2576', 'R2579', 'R2582', 'R2584', 'R2585', 'R2588']


# Potential bugs:
# boosting might need permutations as well
# may need selstop=False in boosting function
# spectrogram computed from each wav file separately and then combined
# - maybe should combine all wavs and then compute spectrogram

boostingFraction = [-0.041,0.21]

basislen = 0.004
partitions = 4


def boost_roi(subdata, roi="cortex", predictors=["carrier", "envelope"], 
              epochrange=list(range(0,10)), outstr="", 
              pcloadf=get_mathlang_carrier, 
              peloadf=get_mathlang_envelope, cocktail=True, permute=False,
              cocktailregion='both'):

    if roi == "cortex":
        ctxll = ['inferiortemporal', 'middletemporal',
                       'superiortemporal','bankssts', 'transversetemporal']
        subctxll = []
    elif roi == "subcortex":
        ctxll = []
        subctxll = ['Brain-Stem', '3rd-Ventricle','Thalamus-Proper', 'VentralDC']
    elif roi == "both":
        ctxll = ['inferiortemporal', 'middletemporal',
                       'superiortemporal','bankssts', 'transversetemporal']
        subctxll = ['Brain-Stem', '3rd-Ventricle','Thalamus-Proper', 'VentralDC']
    else:
        raise Exception('region of interest must be cortex, subcortex, or both')
    
    llA = ['ctx-lh-'+l for l in ctxll] + ['ctx-rh-'+l for l in ctxll] + subctxll + ['Left-'+l for l in subctxll] + ['Right-'+l for l in subctxll]
    
    fwd_sol_file, = (subdata / 'sourcespace').glob("*vol-7*fwd*.fif")
    if not cocktail:
        print(list((subdata).glob("*single*epo*.fif")))
        epo_file, = (subdata).glob("*single*epo*.fif")
        cov_file, = (subdata / 'sourcespace').glob("*single*cov*.fif")
    else:
        print(list((subdata).glob("*cocktail*epo*.fif")))
        epo_file, = (subdata).glob("*cocktail*epo*.fif")
        cov_file, = (subdata / 'sourcespace').glob("*cocktail*cov*.fif")
    
    print("preprocessing epochs")
    epochs = mne.read_epochs(epo_file)
    epochs = epochs[epochrange]
    epochs.crop(0, 18, False)
    
    print("read forward solution and computing inverse operator")
    fwd = mne.read_forward_solution(fwd_sol_file)
    cov = mne.read_cov(cov_file)
    evoked = epochs.average()
    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, cov, loose=1)
    
    print("applying inverse operator to epochs")
    stc = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=0.111, pick_ori='vector')
    src = inv['src']

    print("concatenating epochs and adding parcellation")
    snd = eel.load.fiff.stc_ndvar(stc, subdata.name, 'vol-7-cortex_brainstem_full', subjects_dir=mri_dir)
    snd.source.parc = eel.load.unpickle(parcfile)
    voilist = [s for s in llA if s in snd.source.parc]
    snd = snd.sub(source=voilist)

    if cocktail:
        if cocktailregion == 'both':
            # do nothing
            pass
        elif cocktailregion == 'middle':
            snd = snd.sub(time=(4.5, 4.5+9))
        elif cocktailregion == 'outer':
            first = snd.sub(time=(0, 4.5))
            last = snd.sub(time=(18-4.5, 18))
            snd = eel.concatenate([first, last])
        else:
            raise Exception(f"cocktailregion got unexpected argument {cocktailregion}")
    snd_aud = eel.concatenate([snd[i:i+1] for i in range(len(epochs))], dim='time')

    print("creating dataset and generating predictors")
    ds = eel.Dataset()

    predictorlist = []

    # brain signals are low-frequency, generally, so we don't really have to
    # worry much about aliasing like we did in the case of preprocessing the
    # carrier predictor
    ds['source'] = eel.resample(snd_aud, 500)

    print("source", ds['source'])


    if "carrier" in predictors: 
        # TODO resample is a no-op
        if cocktail:
            ds['carrier_fg'] = eel.resample(pcloadf(single_speaker=False, foreground=True), 500)
            ds['carrier_bg'] = eel.resample(pcloadf(single_speaker=False, foreground=False), 500)

            predictorlist.append('carrier_fg')
            predictorlist.append('carrier_bg')
        else:
            ds['carrier'] = eel.resample(pcloadf(single_speaker=True), 500)
            print("carrier", ds['carrier'])
    if "envelope" in predictors:
        # resample should not affect aliasing since we have already filtered to
        # the high-gamma range (70-200 Hz)
        if cocktail:
            ds['envelope_fg'] = eel.resample(peloadf(single_speaker=False, foreground=True), 500)
            ds['envelope_bg'] = eel.resample(peloadf(single_speaker=False, foreground=False), 500)

            predictorlist.append('envelope_fg')
            predictorlist.append('envelope_bg')
        else:
            ds['envelope'] = eel.resample(peloadf(single_speaker=True), 500)
            print("envelope", ds['envelope'])

    
            
    if len(predictorlist) == 0:
        predictorlist = predictors

        
    print("boosting")
    resM, _ = boostWrap(ds, predictorlist, outpath, subdata.name, 
                           f'vol-7-{roi}{outstr}', boostingFraction, basislen, 
                           partitions, 'source', False)

    if not permute:
        return resM

    print("boosting permuted predictors")
    nperm = 3
    for pred in predictorlist:
        ds = permutePred(ds, pred, nperm)

    for npr in range(nperm):
        _, _ = boostWrap(ds, [i+'_p'+str(npr) for i in predictorlist], 
                           outpath, subdata.name, f'vol-7-{roi}{outstr}', 
                           boostingFraction, basislen, partitions, 'source', 
                           False)
    
    return resM
