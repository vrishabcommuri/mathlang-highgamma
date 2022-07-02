from context import mathlang
from mathlang import *
from mathlang.trf import *

import eelbrain as eel

# from multiprocessing import Pool

def boost_subject(subject_dir):
    subject_id = subject_dir.name
    print("processing subject", subject_id)

    if subject_id in first_group_subs:
        resM_cortex = boost_roi(subject_dir, "cortex", ["envelope", "carrier"], 
                                epochrange=list(range(20,30)),
                                outstr="-testsskateMath", 
                                pcloadf=get_kate_math_carrier,
                                peloadf=get_kate_math_envelope, cocktail=False,
                                permute=True)

    else:
        resM_cortex = boost_roi(subject_dir, "cortex", ["envelope", "carrier"], 
                                epochrange=list(range(0,10)),
                                outstr="-testsskateMath", 
                                pcloadf=get_kate_math_carrier,
                                peloadf=get_kate_math_envelope, cocktail=False,
                                permute=True)


if __name__ == '__main__':
    subject_directories = list(subdatadir.glob("R*"))

    for i in subject_directories:
        if i.name == "R2566":
            boost_subject(i)
    
            
