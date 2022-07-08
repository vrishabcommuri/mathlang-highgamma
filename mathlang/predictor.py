from functools import lru_cache
import eelbrain as eel
import scipy
import numpy as np
from .config import *


SINGLE_SPEAKER_STIMFOLDER = "single_speaker_stimuli"
COCKTAIL_STIMFOLDER = "cocktail_party_stimuli"


SINGLE_SPEAKER_NUMTRIALS = 10 # 10 trials per condition (e.g., 10 for JamesMATH)
SINGLE_SPEAKER_TRIAL_LEN = 18 # 18 seconds long

# 6 trials per condition (e.g., 6 for JamesMATH foreground)
COCKTAIL_NUMTRIALS = 6
COCKTAIL_TRIAL_LEN = 18 # 18 seconds long


def load_single_wav(trialnum, stimdir, condition):
    # trial numbers in files are given in the form trial00, trial01, etc. this
    # simply converts an integer, say, int(1) into a string of the form "01"
    trialnum = str(trialnum).zfill(2)

    trialprefix = ""
    if stimdir == COCKTAIL_STIMFOLDER:
        trialprefix = condition[-2:] + "_"
        condition = condition[:-3]

    stim_file = datadir / f"stimuli/{stimdir}/{condition}/{trialprefix}trial{str(trialnum).zfill(2)}.wav"

    wav = eel.load.wav(stim_file)
    wav.x = wav.x.astype(np.float64)

    # ensures no more than 18 secs worth of data (sometimes there may be a few
    # extra samples)
    wav = wav.sub(time=(0,18))      
    fs = wav.info['samplingrate']

    # all analyses are carried out at 500 Hz
    fsfilt = 500
    decimation_factor = fs/fsfilt
    
    # # we want to resample to 500 Hz from 22.05 kHz, so the decimation factor is
    # # 44.1, but scipy's decimate only supports integer decimation factors. to
    # # get around this, we upsample by a factor of 10 and then downsample by a
    # # factor of 441.
    # wav = eel.resample(wav, fs*10)

    # # we could, nominally, use eel.resample to downsample, but after inspecting
    # # the eel.resample code
    # # (https://github.com/christianbrodbeck/Eelbrain/blob/master/eelbrain/_ndvar.py)
    # # it appears that eel.resample is a simple wrapper for scipy.signal.resample
    # # which DOES NOT apply an anti-aliasing pre-filter before downsampling. I am
    # # not sure what the implications of downsampling from 22.05 kHz (the
    # # frequency of the stimulus wav files) to 1kHz as was done in the original
    # # high gamma code
    # # (https://github.com/jpkulasingham/highfreqTRF/blob/master/make_predictors.py)
    # # are, but I feel that the effects will be non-negligible. hence, I break
    # # apart the NDVar and decimate the signal appropriately before running the
    # # analyses.
    # x = scipy.signal.decimate(wav.x, int(decimation_factor*10)) 


    x = scipy.signal.decimate(wav.x, int(decimation_factor))
    fstrue = fs/int(decimation_factor)

    wav = eel.NDVar(x, eel.UTS(0, 1/fstrue, len(x)))
    wav = eel.resample(wav, fsfilt)
    wav = wav.sub(time=(0,18))

    return wav


def load_single_spectrogram(trialnum, stimdir, condition):
    # trial numbers in files are given in the form trial00, trial01, etc. this
    # simply converts an integer, say, int(1) into a string of the form "01"
    trialnum = str(trialnum).zfill(2)

    trialsuffix = "spec"
    trialprefix = ""
    if stimdir == COCKTAIL_STIMFOLDER:
        # ugly hack. TODO fix this
        trialprefix = condition[-2:] + "_"
        trialsuffix = ""

    # spectrograms are precomputed (see ./doc/design.md for info on how to do
    # this) so we just load them from the directory. 
    mat = scipy.io.loadmat(datadir / f"stimuli_matfiles/{stimdir}/{condition}/{trialprefix}trial{trialnum}{trialsuffix}.mat")

    # get all the spectrogram info
    f, fs, Sxx = mat['frequencies'].flatten(), mat['samplingrate'].flatten(), \
                 mat['specgram']

    Sxx = Sxx.astype(np.float64)

    return f, fs, Sxx


def get_carrier(loadfunc=load_single_wav, 
                numsec=SINGLE_SPEAKER_TRIAL_LEN * SINGLE_SPEAKER_NUMTRIALS, 
                *args):
    FILT_KWARGS = {
        'filter_length': 'auto',
        'method': 'fir',
        'fir_design': 'firwin',
        'l_trans_bandwidth': 5, 
        'h_trans_bandwidth': 5,
    }
    
    # TODO remove this comment if not necessary
    # the original paper used a sampling frequency of 1000 for creating the
    # envelope and then downsampled this just before boosting. this probably
    # doesn't make any difference, but I have preserved the original behavior
    # here to avoid any bugs.
    fsfilt = 500 # TODO was 1000

    # filter high-gamma range 70-200 Hz
    filtc = [70, 200]
    
    # get wav file for ith trial in specified condition (e.g.,  
    # *args = [1, "single_speaker_stimuli", "01jamesMath"] indicates first trial
    # in james math condition) 
    wav = loadfunc(*args)
    
    # filter to high gamma range
    carrier = eel.filter_data(wav, filtc[0], filtc[1], **FILT_KWARGS)
    
    # now that the data are filtered we can resample the signal to 500 Hz (this
    # is done in boost_roi in ./trf.py) without aliasing since this is above
    # nyquist
    
    # standardize carrier
    carrier -= carrier.mean()
    carrier /= carrier.std()

    # reformat NDVar so that it has a Case dimension -- this is required for the
    # boosting code.
    carrier = eel.NDVar(carrier.x.reshape(1,-1), (eel.Case, carrier.time))
    # TODO this should not be hard coded
    # cocktailregion = 'outer'
    cocktailregion = 'both'
    if cocktailregion == 'both':
        # do nothing
        pass
    elif cocktailregion == 'middle':
        carrier = carrier.sub(time=(4.5, 4.5+9))
    elif cocktailregion == 'outer':
        carrier = eel.concatenate([carrier.sub(time=(0, 4.5)), 
                                   carrier.sub(time=(18-4.5, 18))])
    else:
        raise Exception(f"cocktailregion got unexpected argument {cocktailregion}")

    return carrier


def get_envelope(loadfunc=load_single_spectrogram, numsec=180, *args):
    FILT_KWARGS = {
        'filter_length': 'auto',
        'method': 'fir',
        'fir_design': 'firwin',
        'l_trans_bandwidth': 5,
        'h_trans_bandwidth': 5,
    }

    # the original paper used a sampling frequency of 1000 for creating the
    # envelope and then downsampled this just before boosting. this probably
    # doesn't make any difference, but I have preserved the original behavior
    # here to avoid any bugs.
    fsfilt = 1000

    # filter high-gamma range 70-200 Hz
    filtc = [70, 200]

    # high-frequency predictor takes frequencies from spectrogram 300 - 4000 Hz
    specrange = (300, 4000)
    
    # get spectrogram for ith trial in specified condition (e.g.,  
    # *args = [1, "single_speaker_stimuli", "01jamesMath"] indicates first trial
    # in james math condition) 
    f, fs, Sxx = loadfunc(*args)

    # Set up an NDVar from the spectrogram (Sxx) and frequencies (f)
    freqs = eel.Scalar('frequency', f, 'Hz')
    spec = eel.NDVar(Sxx, dims=(eel.UTS(0, 1/fs, len(Sxx)), freqs))
    
    # grab the appropriate frequency range for high-frequency predictor
    spec = spec.sub(frequency=specrange)

    # sometimes the spectrogram is a couple of samples too long -- this will
    # drop any extra samples (should only be two or three samples max -- you can
    # delete this line to check the predictor length via error message when
    # eelbrain complains)
    spec = spec.sub(time=(0, numsec))

    # filter the predictor signal to the high-gamma range
    specfilt = eel.filter_data(spec, filtc[0], filtc[1], **FILT_KWARGS)

    # average across frequency bins to obtain a 1-D time series
    filt = specfilt.mean('frequency')
    hfe = eel.filter_data(filt, filtc[0], filtc[1], **FILT_KWARGS)

    # standardize the predictor
    hfe -= hfe.mean()
    hfe /= hfe.std()
    
    # cast to an eelbrain ndvar
    time = eel.UTS(0, 1/fsfilt, (numsec)*fsfilt) 
    predictor = eel.NDVar(hfe.x.reshape(1,-1), (eel.Case, time))
    # TODO this should not be hard coded
    # cocktailregion = 'outer'
    cocktailregion = 'both'
    if cocktailregion == 'both':
        # do nothing
        pass
    elif cocktailregion == 'middle':
        predictor = predictor.sub(time=(4.5, 4.5+9))
    elif cocktailregion == 'outer':
        predictor = eel.concatenate([predictor.sub(time=(0, 4.5)), 
                                     predictor.sub(time=(18-4.5, 18))])
    else:
        raise Exception(f"cocktailregion got unexpected argument {cocktailregion}")
    return predictor

@lru_cache(maxsize=None)
def combine_n_carriers(loadfunc, conditions, stimdir=SINGLE_SPEAKER_STIMFOLDER, 
                       stimlen=SINGLE_SPEAKER_TRIAL_LEN, 
                       numtrials=SINGLE_SPEAKER_NUMTRIALS):
    if isinstance(conditions, str):
        conditions = [conditions]

    carrier = None
    for condition in conditions:
        for i in range(numtrials):
            print(loadfunc, stimlen, i, stimdir, condition)
            c = get_carrier(loadfunc, stimlen, i, stimdir, condition)
            if carrier is None:
                carrier = c
            else:
                carrier = eel.concatenate([carrier, c], dim='time')
    return carrier


@lru_cache(maxsize=None)
def combine_n_envelopes(loadfunc, conditions, stimdir=SINGLE_SPEAKER_STIMFOLDER, 
                        stimlen=SINGLE_SPEAKER_TRIAL_LEN, 
                        numtrials=SINGLE_SPEAKER_NUMTRIALS):
    if isinstance(conditions, str):
        conditions = [conditions]
        
    envelope = None
    for condition in conditions:
        for i in range(numtrials):
            e = get_envelope(loadfunc, stimlen, i, stimdir, condition)
            if envelope is None:
                envelope = e
            else:
                envelope = eel.concatenate([envelope, e], dim='time')
    return envelope


################################################################################
# carrier wrapper functions
################################################################################
@lru_cache(maxsize=None)
def get_james_math_carrier(single_speaker=True, foreground=True):
    if single_speaker:
        math_carrier = combine_n_carriers(load_single_wav,
                                        "01jamesMATH",
                                        SINGLE_SPEAKER_STIMFOLDER,
                                        SINGLE_SPEAKER_TRIAL_LEN,
                                        SINGLE_SPEAKER_NUMTRIALS)

    else:
        if foreground:
            math_carrier_first = combine_n_carriers(load_single_wav,
                                            "08jamesMATH_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            math_carrier_second = combine_n_carriers(load_single_wav,
                                            "12jamesMATH_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)
        else:
            math_carrier_first = combine_n_carriers(load_single_wav,
                                            "08jamesMATH_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            math_carrier_second = combine_n_carriers(load_single_wav,
                                            "12jamesMATH_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

        math_carrier = eel.concatenate([math_carrier_first, math_carrier_second], dim='time')

    return math_carrier


@lru_cache(maxsize=None)
def get_james_lang_carrier(single_speaker=True, foreground=True):
    if single_speaker:
        lang_carrier = combine_n_carriers(load_single_wav,
                                        "02jamesLANG",
                                        SINGLE_SPEAKER_STIMFOLDER,
                                        SINGLE_SPEAKER_TRIAL_LEN,
                                        SINGLE_SPEAKER_NUMTRIALS)

    else:
        if foreground:
            lang_carrier_first = combine_n_carriers(load_single_wav,
                                            "07jamesLANG_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            lang_carrier_second = combine_n_carriers(load_single_wav,
                                            "11jamesLANG_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)
        else:
            lang_carrier_first = combine_n_carriers(load_single_wav,
                                            "07jamesLANG_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            lang_carrier_second = combine_n_carriers(load_single_wav,
                                            "11jamesLANG_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

        lang_carrier = eel.concatenate([lang_carrier_first, lang_carrier_second], dim='time')

    return lang_carrier

@lru_cache(maxsize=None)
def get_kate_math_carrier(single_speaker=True, foreground=True):
    if single_speaker:
        math_carrier = combine_n_carriers(load_single_wav,
                                        "03kateMATH",
                                        SINGLE_SPEAKER_STIMFOLDER,
                                        SINGLE_SPEAKER_TRIAL_LEN,
                                        SINGLE_SPEAKER_NUMTRIALS)

    else:
        if foreground:
            math_carrier_first = combine_n_carriers(load_single_wav,
                                            "06kateMATH_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            math_carrier_second = combine_n_carriers(load_single_wav,
                                            "10kateMATH_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)
        else:
            math_carrier_first = combine_n_carriers(load_single_wav,
                                            "06kateMATH_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            math_carrier_second = combine_n_carriers(load_single_wav,
                                            "10kateMATH_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

        math_carrier = eel.concatenate([math_carrier_first, math_carrier_second], dim='time')

    return math_carrier

@lru_cache(maxsize=None)
def get_kate_lang_carrier(single_speaker=True, foreground=True):
    if single_speaker:
        lang_carrier = combine_n_carriers(load_single_wav,
                                        "04kateLANG",
                                        SINGLE_SPEAKER_STIMFOLDER,
                                        SINGLE_SPEAKER_TRIAL_LEN,
                                        SINGLE_SPEAKER_NUMTRIALS)

    else:
        if foreground:
            lang_carrier_first = combine_n_carriers(load_single_wav,
                                            "05kateLANG_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            lang_carrier_second = combine_n_carriers(load_single_wav,
                                            "09kateLANG_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)
        else:
            lang_carrier_first = combine_n_carriers(load_single_wav,
                                            "05kateLANG_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            lang_carrier_second = combine_n_carriers(load_single_wav,
                                            "09kateLANG_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

        lang_carrier = eel.concatenate([lang_carrier_first, lang_carrier_second], dim='time')

    return lang_carrier


@lru_cache(maxsize=None)
def get_mathlang_carrier(single_speaker = True):
    if single_speaker:
        math_carrier = combine_n_carriers(load_single_wav,
                                        "01jamesMATH",
                                        SINGLE_SPEAKER_STIMFOLDER,
                                        SINGLE_SPEAKER_TRIAL_LEN,
                                        SINGLE_SPEAKER_NUMTRIALS)

        lang_carrier = combine_n_carriers(load_single_wav, 
                                        "02jamesLANG",
                                        SINGLE_SPEAKER_STIMFOLDER,
                                        SINGLE_SPEAKER_TRIAL_LEN,
                                        SINGLE_SPEAKER_NUMTRIALS)
    
    else:
        raise Exception("Not implemented yet")

        # math_carrier = combine_n_carriers(# TODO implement me, 
        #                                 COCKTAIL_TRIAL_LEN,
        #                                 COCKTAIL_NUMTRIALS)

        # lang_carrier = combine_n_carriers(# TODO implement me, 
        #                                 COCKTAIL_TRIAL_LEN,
        #                                 COCKTAIL_NUMTRIALS)

    carrier = eel.concatenate([math_carrier, lang_carrier], dim='time')
    return carrier

@lru_cache(maxsize=None)
def get_james_math_envelope(single_speaker=True, foreground=True):
    if single_speaker:
        math_envelope = combine_n_envelopes(load_single_spectrogram,
                                        "01jamesMATH",
                                        SINGLE_SPEAKER_STIMFOLDER,
                                        SINGLE_SPEAKER_TRIAL_LEN,
                                        SINGLE_SPEAKER_NUMTRIALS)

    else:
        if foreground:
            math_envelope_first = combine_n_envelopes(load_single_spectrogram,
                                            "08jamesMATH_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            math_envelope_second = combine_n_envelopes(load_single_spectrogram,
                                            "12jamesMATH_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)
        else:
            math_envelope_first = combine_n_envelopes(load_single_spectrogram,
                                            "08jamesMATH_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            math_envelope_second = combine_n_envelopes(load_single_spectrogram,
                                            "12jamesMATH_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

        math_envelope = eel.concatenate([math_envelope_first, math_envelope_second], dim='time')

    return math_envelope

@lru_cache(maxsize=None)
def get_james_lang_envelope(single_speaker=True, foreground=True):
    if single_speaker:
        lang_envelope = combine_n_envelopes(load_single_spectrogram,
                                        "02jamesLANG",
                                        SINGLE_SPEAKER_STIMFOLDER,
                                        SINGLE_SPEAKER_TRIAL_LEN,
                                        SINGLE_SPEAKER_NUMTRIALS)

    else:
        if foreground:
            lang_envelope_first = combine_n_envelopes(load_single_spectrogram,
                                            "07jamesLANG_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            lang_envelope_second = combine_n_envelopes(load_single_spectrogram,
                                            "11jamesLANG_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)
        else:
            lang_envelope_first = combine_n_envelopes(load_single_spectrogram,
                                            "07jamesLANG_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            lang_envelope_second = combine_n_envelopes(load_single_spectrogram,
                                            "11jamesLANG_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

        lang_envelope = eel.concatenate([lang_envelope_first, lang_envelope_second], dim='time')

    return lang_envelope


@lru_cache(maxsize=None)
def get_kate_math_envelope(single_speaker=True, foreground=True):
    if single_speaker:
        math_envelope = combine_n_envelopes(load_single_spectrogram,
                                        "03kateMATH",
                                        SINGLE_SPEAKER_STIMFOLDER,
                                        SINGLE_SPEAKER_TRIAL_LEN,
                                        SINGLE_SPEAKER_NUMTRIALS)

    else:
        if foreground:
            math_envelope_first = combine_n_envelopes(load_single_spectrogram,
                                            "06kateMATH_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            math_envelope_second = combine_n_envelopes(load_single_spectrogram,
                                            "10kateMATH_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)
        else:
            math_envelope_first = combine_n_envelopes(load_single_spectrogram,
                                            "06kateMATH_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            math_envelope_second = combine_n_envelopes(load_single_spectrogram,
                                            "10kateMATH_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

        math_envelope = eel.concatenate([math_envelope_first, math_envelope_second], dim='time')

    return math_envelope

@lru_cache(maxsize=None)
def get_kate_lang_envelope(single_speaker=True, foreground=True):
    if single_speaker:
        lang_envelope = combine_n_envelopes(load_single_spectrogram,
                                        "04kateLANG",
                                        SINGLE_SPEAKER_STIMFOLDER,
                                        SINGLE_SPEAKER_TRIAL_LEN,
                                        SINGLE_SPEAKER_NUMTRIALS)

    else:
        if foreground:
            lang_envelope_first = combine_n_envelopes(load_single_spectrogram,
                                            "05kateLANG_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            lang_envelope_second = combine_n_envelopes(load_single_spectrogram,
                                            "09kateLANG_fg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)
        else:
            lang_envelope_first = combine_n_envelopes(load_single_spectrogram,
                                            "05kateLANG_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

            lang_envelope_second = combine_n_envelopes(load_single_spectrogram,
                                            "09kateLANG_bg",
                                            COCKTAIL_STIMFOLDER,
                                            COCKTAIL_TRIAL_LEN,
                                            COCKTAIL_NUMTRIALS)

        lang_envelope = eel.concatenate([lang_envelope_first, lang_envelope_second], dim='time')

    return lang_envelope


@lru_cache(maxsize=None)
def get_james_mathlang_envelope(single_speaker=True, foreground=True):
    math_envelope = get_james_math_envelope(single_speaker, foreground)
    lang_envelope = get_james_lang_envelope(single_speaker, foreground)
    
    envelope = eel.concatenate([math_envelope, lang_envelope], dim='time')
    return envelope


def _write_res(res: eel.BoostingResult, output_folder: str, subject: str, predstr: str, outstr: str, modelstr: str):
    print(f'{subject} {predstr} {outstr} {modelstr} r max = {res.r.max():.4g}\n')
    print(f'{subject} {predstr} {outstr} {modelstr} max = {res.h.max():.4g}\n')
    with open(f'{output_folder}/Source/boosting.txt', 'a+') as f:
        f.write(f'{subject} {predstr} {outstr} {modelstr} r max = {res.r.max():.4g}\n')
        f.write(f'{subject} {predstr} {outstr} {modelstr} max = {res.h.max():.4g}\n')

def permutePred(ds, predstr, nperm=2):
    """Permutes the given predictor""" 
    for npr in range(0,nperm):
        xnd = ds[predstr].copy()
        if ds[predstr].has_case:
            for j in range(0,len(ds[predstr])):
                aa = np.roll(ds[predstr][j].x, int((npr+1) * len(xnd[j]) / (nperm+1)))
                xnd[j] = eel.NDVar(aa,dims=xnd[j].dims,name=predstr+'_p'+str(npr))
        else:
            xnd.x = np.roll(xnd.x, int((npr + 1) * len(xnd) / (nperm + 1)))
        xnd.name = predstr+'_p'+str(npr)
        ds[xnd.name] = xnd
    return ds
        

def boostWrap(ds: eel.Dataset, predlist: list, output_folder: str, subject: str, outstr: str, bFrac: list, basislen: float, partitions: int, rstr: str = 'source', permflag: bool = True):
    """Boosts the given predictor. If permflag == True, boosts permuted model also"""
    resM = eel.boosting(ds[rstr], [ds[predstr] for predstr in predlist], bFrac[0], bFrac[1], basis=basislen, partitions=partitions, selective_stopping=False) # selstop = False added 11/9/21
#     _write_res(resM, output_folder, subject, predlist, outstr, modelstr='Model')
    resN = None
#     if permflag:
#         resN = eel.boosting(ds[rstr], ds[predstr + '_p0'], bFrac[0], bFrac[1], basis=basislen, partitions=partitions)
#         _write_res(resN, output_folder, subject, predstr, outstr, modelstr='Noise')
#         eel.save.pickle([resM,resN],f'{output_folder}/Source/Pickle/{subject}_{predstr}{outstr}.pkl')
#     else:
#         eel.save.pickle(resM,f'{output_folder}/Source/Pickle/{subject}_{predstr}{outstr}.pkl')
    eel.save.pickle(resM,f'{output_folder}/Source/Pickle/{subject}_{"-".join(predlist)}{outstr}.pkl')
    # resN = f'{output_folder}/Source/Pickle/{subject}_{"-".join(predlist)}{outstr}.pkl'
    return resM, resN


def res2meas(res, roi, pred, scale=True):
    out = []
    for i in range(len(res)):
        if roi == 'cortex':
            if pred == 'envelope':
                if scale:
                    out.append(res[i][1].h_scaled[1].norm('space').x)
                else:
                    out.append(res[i][1].h[1].norm('space').x)
            else:
                if scale:
                    out.append(res[i][1].h_scaled[0].norm('space').x)
                else:
                    out.append(res[i][1].h[0].norm('space').x)
        else:
            if pred == 'envelope':
                if scale:
                    out.append(res[i][2].h_scaled[1].norm('space').x)
                else:
                    out.append(res[i][2].h[1].norm('space').x)
            else:
                if scale:
                    out.append(res[i][2].h_scaled[0].norm('space').x)
                else:
                    out.append(res[i][2].h[0].norm('space').x)
    return np.array(out)