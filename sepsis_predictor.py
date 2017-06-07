from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import math

class sepsis_predictor:
    def __init__(self, predict_ahead = 0, lookback = 5):
        """
        - predict_ahead is how many hours ahead of most recent column of data to predict sepsis (0 is diagnose the current col)
        - self.lookback is how many hours to include in the prediction (1 means just use this column)
        """

        assert predict_ahead >= 0 and lookback >= 1

        self.lookback = lookback
        self.predict_ahead = predict_ahead
        self.scaler = None
        self.classifier = None
        

    def _init_scaler(self, data):
        """Just scales, doesn't know anything about the data"""
        self.scaler = preprocessing.StandardScaler().fit(data)


    def _scale(self, data):

        return self.scaler.transform(data)


    def _balance(self, X, y, percent_septic):
        """Remove non-sepsis data at random to make sepsis cases 50% of the data"""
        #split X on sepsis condition
        no_sepsis = X[y == 0]
        sepsis = X[y == 1]

        #subsample no_sepsis
        no_sepsis_sample = no_sepsis[np.random.choice(no_sepsis.shape[0], int(float(sepsis.shape[0])/percent_septic - sepsis.shape[0]), replace = False)]

        #attach y to X for shuffling
        sepsis           = np.append(sepsis,           np.ones([sepsis.shape[0],1]),            axis = 1)
        no_sepsis_sample = np.append(no_sepsis_sample, np.zeros([no_sepsis_sample.shape[0],1]), axis = 1)
        balanced_data = np.concatenate((sepsis, no_sepsis_sample))
        np.random.shuffle(balanced_data)

        #and split out again
        return balanced_data[:,:-1], balanced_data[:,-1]

    
    def _digest_for_training(self, patients, pre_sepsis_ignore, sepsis_length):
        """
        Pick out times to train on and format them for training
        - pre_sepsis_ignore is how many hours before the data says a patient has sepsis to remit from the training set (this is 
            inspired by the thought that some sepsis may be caught an hour or so late)
        - sepsis_length is how many hours after septic is condition is noticed to count as septic.
        """
        X = [] #input
        y = [] #output

        for patient in tqdm(patients.itervalues(), total = len(patients)):

            #PICK COLUMNS TO KEEP:
            #Rows here are time steps, remember!
            total_rows = patient['measurements'].shape[0]
            if math.isnan(patient['column_onset']): #if patient does not contract sepsis during visit, take all rows
                training_rows = range(self.lookback, total_rows) 
            else: #if patient *does* contract sepsis, take...
                #...the healthy portion, less a pre-sepsis buffer...
                training_rows = list(range(self.lookback, int(patient['column_onset']) - pre_sepsis_ignore - 1))
                #...as well as some samples during the septic period:
                training_rows += list(range(max(self.lookback-1, int(patient['column_onset'])), \
                                            min(int(patient['column_onset'])+sepsis_length,total_rows)))

            #MAP TIME SERIES TO SEPARATE ROWS
            for i in training_rows:
                # assert i >= self.lookback
                if i < self.lookback: continue

                X.append(self._patient_to_features(patient, i))
                y.append(int((i + self.predict_ahead) >= patient['column_onset']))

        return np.asarray(X), np.asarray(y)
    

    def _patient_to_features(self, patient, time):
        """
        Given a single patient and a row of time, map data to an input row
        appropriate for clf.

        - time is 0-justified.
        - 'column_onset' is confusing--it means rows in this data format.
        - the time here is the last datapoint on which to do prediction
        """
        datum = [patient['weight'], patient['gender'], patient['age']]
        for mmt in patient['measurements'].T:
            for i in range(self.lookback):
                # if there isn't enough historical data, copy it backwards
                datum.append(mmt[max(time - i, 0)])

        return datum


    def train(self, patients, pre_sepsis_ignore = 1, sepsis_length = 1, percent_septic = 0.3):
        """
        Standard scaling of data to prep for learning
        Input is a dict of patients with numpy array of measurements, 
        rows time and columns measurements

        - pre_sepsis_ignore is number of pre-sepsis hours to leave out, with the 
        thought that the patient may have sepsis at that time and it just wasn't
        diagnosed yet.
        - sepsis_length is number of hours after column_onset to treat as having sepsis
        """
        assert pre_sepsis_ignore >= 0 and sepsis_length >= 1

        #map into rows of patient-hours
        X_raw, y_raw = self._digest_for_training(patients, pre_sepsis_ignore, sepsis_length)

        #make 50% septic
        X_balanced, y = self._balance(X_raw, y_raw, percent_septic)

        #normalize
        self._init_scaler(X_balanced)
        X = self._scale(X_balanced)

        #train
        # self.classifier = RandomForestClassifier(n_estimators = 200,
        #                              min_samples_split = 2,
        #                              random_state = 0,
        #                              n_jobs = -1).fit(X, y)

        # self.classifier = MLPClassifier(hidden_layer_sizes = (20,10)).fit(X, y)

        self.classifier = GradientBoostingClassifier(n_estimators = 100,
                                         learning_rate = .6,
                                         max_depth = 1).fit(X, y)


    def predict(self, patient):
        """
        Takes patient data, formats it for predictor, runs prediction on last data
        """
        #grab tail of patient record
        x_raw = np.asarray(self._patient_to_features(patient, patient['measurements'].shape[0] - 1)).reshape(1,-1)
        
        #scale
        x = self._scale(x_raw)

        #predict
        return bool(self.classifier.predict(x)[0])