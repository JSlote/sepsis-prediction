# Imputation Notes:
# - 1 = male, 0 = female
# - When weights are unknown, average UK values (in kg) are used
# - When data is missing, we interpolate it with a Piecewise Cubic Hermite Interpolating Polynomial

# For python 2 & 3 compatibility:
# Import future builtins
from __future__ import print_function
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)
# Disallow removed builtins like xrange
from future.builtins.disabled import *

import json
import math
import numpy as np
from scipy.interpolate import PchipInterpolator as interpolate #for imputation
import pickle
from tqdm import tqdm

averages = {
    'male_weight' : 83.6,
    'female_weight': 70.2,
    'SysABP': 120,
    'DiasABP': 80,
    'Temp': 37,
    'HR': 80,
    'SpO2': 97,
    'RespRate': 16,
    'MAP': 83,
    'PP': 57
}

def interpolate_impute(series):
    assert not math.isnan(series[0])
    # x, y for known data
    x = []
    y = []
    # cols for x values to infer from interpolation
    to_impute = []

    for col, datum in enumerate(series):
        if math.isnan(datum):
            to_impute.append(col)
        else:
            x.append(col)
            y.append(datum)

    #if we only have one data point, there's nothing to interpolate, so just copy across
    if len(x) == 1:
        interpolation = lambda z: y[0]
    else:
        interpolation = interpolate(x, y, extrapolate = True)

    imputed = series[:]
    for col in to_impute:
        #we're not overwriting
        assert math.isnan(imputed[col])
        imputed[col] = interpolation(col)

    for col in x:
        #we have left no nans
        assert not math.isnan(imputed[col])

    return imputed

def carryforward_impute(series):
    assert not math.isnan(series[0])
    imputed = []
    for val in series:
        if math.isnan(val):
            imputed.append(imputed[-1])
        else:
            imputed.append(val)

    return imputed

print("Beginning imputation...")
with open('MIMIC3_processed.json') as data_file:
    mimic = json.load(data_file)

print("Loaded up", len(mimic.keys()), "total visits")

# Find which measurements are worth imputing
all_measurements = set()
common_measurements = set(next(mimic.itervalues())['measurements'].keys())

for patient in mimic.itervalues():
    patient_measurements = set(patient['measurements'].keys())
    all_measurements = all_measurements.union(patient_measurements)
    common_measurements = common_measurements.intersection(patient_measurements)

print("All measurements:\n", all_measurements)

common_measurements = sorted(common_measurements) #map to a sorted list to be used later
print("\nCommon measurements to be imputed and saved:\n", common_measurements)

# Imputation

mimic_interpolated = {}
mimic_carryforward = {}

missing_mmts = set()

for patient_name, patient_data in tqdm(mimic.items()):
    interpolated_patient_data = {}
    carryforward_patient_data = {}

    #This dataset records only male/female. 1 for male, 0 for female.
    gender = int(patient_data['gender'] == 'M')
    interpolated_patient_data['gender'], \
    carryforward_patient_data['gender'] = gender, gender


    if math.isnan(patient_data['weight']):
        #average weights by gender, from the ONS
        weight = averages['male_weight'] if patient_data['gender'] else averages['female_weight']
    else:
        weight = patient_data['weight']

    interpolated_patient_data['weight'], \
    carryforward_patient_data['weight'] = weight, weight

    interpolated_patient_data['age'], \
    carryforward_patient_data['age'] = patient_data['age'], patient_data['age']

    interpolated_patient_data['column_onset'], \
    carryforward_patient_data['column_onset'] = patient_data['column_onset'], patient_data['column_onset']

    interpolated_patient_data['label_sepsis'], \
    carryforward_patient_data['label_sepsis'] = patient_data['label_sepsis'], patient_data['label_sepsis']

    interpolated_mmts = []
    carryforward_mmts = []

    for mmt in common_measurements:
        if math.isnan(patient_data['measurements'][mmt][0]):
            patient_data['measurements'][mmt][0] = averages[mmt]

        series = patient_data['measurements'][mmt]

        interpolated_mmts.append(interpolate_impute(series))
        carryforward_mmts.append(carryforward_impute(series))

    interpolated_patient_data['measurements'] = np.asarray(interpolated_mmts).T
    carryforward_patient_data['measurements'] = np.asarray(carryforward_mmts).T

    mimic_interpolated[patient_name] = interpolated_patient_data
    mimic_carryforward[patient_name] = carryforward_patient_data

print("Saving imputed dataset for next time...")
pickle.dump( mimic_interpolated, open( "mimic_interpolated.dat", "wb" ) )
pickle.dump( mimic_carryforward, open( "mimic_carryforward.dat", "wb" ) )
print("Imputation complete.")