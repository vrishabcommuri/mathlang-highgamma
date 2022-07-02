from context import mathlang
from mathlang import *
from mathlang.trf import *

import eelbrain as eel

# from multiprocessing import Pool

def boost_subject(subject_dir):
    subject_id = subject_dir.name
    print("processing subject", subject_id)

    if subject_id in first_group_subs:
        # epochs 1-10 are JamesMATH
        resM_cortex = boost_roi(subject_dir, "cortex", ["envelope", "carrier"], 
                                epochrange=list(range(0,20)),
                                outstr="-testssjamesMathLang", 
                                pcloadf=get_mathlang_carrier,
                                peloadf=get_mathlang_envelope,)

    else:
        # epochs 21-30 are JamesMath (epochs are 0-indexed, so start at 20)
        resM_cortex = boost_roi(subject_dir, "cortex", ["envelope", "carrier"], 
                                epochrange=list(range(20,40)),
                                outstr="-testssjamesMathLang", 
                                pcloadf=get_mathlang_carrier,
                                peloadf=get_mathlang_envelope,)


if __name__ == '__main__':
    subject_directories = list(subdatadir.glob("R*"))

    for i in subject_directories:
        boost_subject(i)
    
            
