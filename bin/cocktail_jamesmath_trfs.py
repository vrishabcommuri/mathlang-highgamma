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
                                epochrange=list(range(18,24)) + list(range(42,48)),
                                outstr="-testcpjamesMath", 
                                pcloadf=get_james_math_carrier,
                                peloadf=get_james_math_envelope, cocktail=True, 
                                cocktailregion='both', permute=True)

    else:
        resM_cortex = boost_roi(subject_dir, "cortex", ["envelope", "carrier"], 
                                epochrange=list(range(6,12)) + list(range(30,36)),
                                outstr="-testcpjamesMath", 
                                pcloadf=get_james_math_carrier,
                                peloadf=get_james_math_envelope, cocktail=True, 
                                cocktailregion='both', permute=True)


if __name__ == '__main__':
    subject_directories = list(subdatadir.glob("R*"))

    for i in subject_directories:
        if i.name == "R2566":
            boost_subject(i)
    
            
