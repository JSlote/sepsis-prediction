# For python 2 & 3 compatibility:
# Import future builtins
from __future__ import print_function
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)
# Disallow removed builtins like xrange
from future.builtins.disabled import *

import pickle

try:
	print("Loading MIMIC Dataset; this could take a while...")
	mimic_interpolated = pickle.load( open( "mimic_interpolated.dat", "rb" ) )
	mimic_carryforward = pickle.load( open( "mimic_carryforward.dat", "rb" ) )
	print("Loading complete")
except:
	print("Couldn't find imputed dataset; generating from 'MIMIC3_processed.json'.")
	print("This WILL take a while.")
	from imputer import mimic_interpolated, mimic_carryforward

# common_measurements = set(next(mimic_nsoa.itervalues())['measurements'].keys())

# for patient in mimic_nsoa.itervalues():
#     patient_measurements = set(patient['measurements'].keys())
#     common_measurements = common_measurements.intersection(patient_measurements)
    
# common_measurements = sorted(common_measurements)mimic