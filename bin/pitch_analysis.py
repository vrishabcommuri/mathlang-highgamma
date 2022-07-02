from context import mathlang
from mathlang import *
import aubio
import scipy
import matplotlib.pyplot as plt


def compute_pitches_from_file(fi):
    s = aubio.source(fi)

    pitch_o = aubio.pitch("yin", samplerate=s.samplerate)

    hop_s = 512
    pitches = []
    confidences = []

    total_frames = 0

    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        pitches += [pitch]
        confidence = pitch_o.get_confidence()
        confidences += [confidence]
       
        total_frames += read
        if read < hop_s: break

    return pitches

def pitches_from_folderlist(flist):
    pitches = []
    for folder in flist:
        for n in range(10):
            trialnum = str(n).zfill(2)
            stimfile = f"{stimdir}{folder}/trial{trialnum}.wav"
            pitches += compute_pitches_from_file(stimfile)
    return pitches

if __name__ == '__main__':
    stimdir = "/Users/vrishabcommuri/Desktop/umd_research/data/gamma_trf/stimuli/single_speaker_stimuli/"
    jamesfolders = ["01jamesMATH", "02jamesLANG"]
    katefolders = ["03kateMATH", "04kateLANG"]

    pitches = pitches_from_folderlist(jamesfolders)

    print(f"james average pitch {scipy.stats.trim_mean(pitches, 0.25):.3f}")

    pitches = pitches_from_folderlist(katefolders)

    print(f"kate average pitch {scipy.stats.trim_mean(pitches, 0.25):.3f}")

    p1 = compute_pitches_from_file("/Users/vrishabcommuri/Downloads/stimuli/Trial1.wav")
    p2 = compute_pitches_from_file("/Users/vrishabcommuri/Downloads/stimuli/Trial2.wav")

    pitches = p1 + p2

    print(f"high gamma average pitch {scipy.stats.trim_mean(pitches, 0.25):.3f}")



# # print(f"Average frequency = {scipy.stats.trim_mean(pitches, 0.25)} hz, skipped {skipped_frames} of {total_frames}")


# fi = "/Users/vrishabcommuri/Desktop/umd_research/data/gamma_trf/stimuli/single_speaker_stimuli/01jamesMATH/trial00.wav"

# # fi = "/Users/vrishabcommuri/Desktop/umd_research/data/gamma_trf/stimuli/single_speaker_stimuli/03kateMATH/trial01.wav"
# # fi = "/Users/vrishabcommuri/Downloads/stimuli/Trial1.wav"