import pathlib 

subdatadir = pathlib.Path.cwd() / "../../../data/gamma_trf/subject_data"
datadir = pathlib.Path.cwd() / "../../../data/gamma_trf/"
parcfile = datadir / "parc_sym-aparc+aseg-nowhite-vol-7-cortex_brainstem_full.pkl"
mri_dir = pathlib.Path.cwd() / "../../../data/gamma_trf/subject_data/fsaverage/mri/"
outpath = pathlib.Path.cwd() / "../outputs/gamma_trfs/"
