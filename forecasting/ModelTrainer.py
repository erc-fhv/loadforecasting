# Import python libraries
#
from tqdm import tqdm
import pickle
import multiprocessing as mp
import Model as model
import pandas as pd
import holidays
import pytz
from demandlib import bdew

# Imports own modules.
# All imports are done relative to the root of the project.
#
import data.weather_data as weather_data
import data.demandprofiles_readout as demandprofiles
import data.standardprofiles_readout as standardprofiles
import LstmAdapter


def optimize_model(model_type, load_profile, pretrain_now, finetune_now):

    # Load a new powerprofile
    with open(load_profile, 'rb') as f:
        (X, Y, lstmAdapter) = pickle.load(f)

    # Train the model
    num_of_features = X['train'].shape[2]
    myModel = model.Model(num_of_features, model_type, lstmAdapter=lstmAdapter)
    history = myModel.train_model(X['train'], Y['train'], X['test'], Y['test'], pretrain_now, finetune_now, verbose=0)
    history = myModel.evaluate(X['test'], Y['test'], history)
    
    # Return the results
    return (model_type, load_profile, history, myModel)

class ModelTrainer:

    def __init__(self, test_config, use_multiprocessing = True):        
        self.use_multiprocessing = use_multiprocessing
        self.test_config = test_config
        self.preprocess(test_config)        

    def preprocess(self):
        
        # Readout the power profiles, bring them to the format needed by the model and store those profiles
        #
        powerProfiles = pd.read_pickle(self.test_config['load_profiles'])

        # Readout the weather data
        #
        startDate = powerProfiles[0].index[0].to_pydatetime().replace(tzinfo=None)
        endDate = powerProfiles[0].index[-1].to_pydatetime().replace(tzinfo=None)
        weather_measurements = weather_data.WeatherMeasurements()
        weatherData = weather_measurements.get_data(
                    startDate = startDate, 
                    endDate = endDate,
                    lat = 51.5085,      # Location:
                    lon = -0.1257,      # London Heathrow,
                    alt = 25,           # Weatherstation   
                    sample_periode = 'hourly', 
                    tz = 'UTC',
                    )
        weatherData = weatherData.loc[:, (weatherData != 0).any(axis=0)]    # remove empty columns

        # Load the public holiday calendar
        public_holidays_dict = holidays.CountryHoliday('GB', prov='ENG', years=range(startDate.year, endDate.year + 1))
        public_holidays_timestamps = [pd.Timestamp(date, tzinfo=pytz.utc) for date in public_holidays_dict.keys()]

        # Bring the power profiles to the model input shape (nr_of_batches, timesteps, features)
        #
        for i, powerProfile in enumerate(powerProfiles[:self.test_config['num_of_communities']]):
            
            # Preprocess data to get X and Y for the model
            out_filename = 'outputs/file_' + str(i) + '.pkl'
            lstmAdapter = LstmAdapter.LstmAdapter(public_holidays_timestamps, train_size = 466,
                                                    addLaggedPower=True, shuffle_data=False)

            X, Y = lstmAdapter.transformData(powerProfile, weatherData)
            with open(out_filename, 'wb') as file:
                pickle.dump((X, Y, lstmAdapter), file)        
        
        # If required, do pretraing
        if self.test_config['pretraining_mode'] == 'store_and_load_weights' \
            or self.test_config['pretraining_mode'] == 'store_weights':

            # Load the BDEW standard load profiles for the years of interest
            standard_loadprofiles = []
            for year in range(startDate.year, endDate.year + 1):
                load_profile = bdew.ElecSlp(year, holidays=public_holidays_timestamps).get_profile({"h0": 1000})
                standard_loadprofiles.append(load_profile)
            
            # Filter the DataFrame for the desired datetime range
            all_standard_loadprofiles = pd.concat(standard_loadprofiles)['h0']
            all_standard_loadprofiles = all_standard_loadprofiles[(all_standard_loadprofiles.index >= startDate) & (all_standard_loadprofiles.index <= endDate)]
            
            # Preprocess data to get X and Y for the model
            self.pretraining_filename = 'outputs/standard_loadprofile.pkl'   
            lstmAdapter = LstmAdapter.LstmAdapter(public_holidays_timestamps, train_size = len(all_standard_loadprofiles), dev_size = 0, 
                                                    add_tda_features=False, addLaggedPower=True, shuffle_data=False,
                                                    use_persistent_entropy = False, seed=0)
            X_model_pretrain, Y_model_pretrain = lstmAdapter.transformData(all_standard_loadprofiles, None)
            with open(self.pretraining_filename, 'wb') as file:
                pickle.dump((X_model_pretrain, Y_model_pretrain, lstmAdapter), file)    
            
            train_histories, myModels = self.train(pretrain_now = True, finetune_now = False)

    def optimize_model_wrapper(self, args):
        return optimize_model(*args)

    def train(self, pretrain_now = False, finetune_now = True):
        
        # Get the load profiles for the training
        if pretrain_now == True:
            data = self.pretraining_filename
        else:
            data = ['outputs/file_' + str(i) + '.pkl' for i in range(self.test_config['num_of_communities'])]

        if self.use_multiprocessing:
            with mp.Pool() as pool:
                results = list(
                    tqdm(
                        pool.imap(self.optimize_model_wrapper, 
                                  [(model_type, load_profile, pretrain_now, finetune_now)
                                  for model_type in self.test_config['models']
                                  for load_profile in data]),
                        total=len(self.test_config['models'])*len(data),
                    )
                )
                pool.close()
                pool.join()

        else:   # Single Process
            results = []
            for model_type in tqdm(self.test_config['models']):
                for load_profile in tqdm(data):
                    result = optimize_model(model_type, load_profile, pretrain_now, finetune_now)
                    results.append(result)

        # create a dict out of the results
        model_types, load_profiles, histories, returnedModels = zip(*results)
        train_histories = dict(sorted(({(model_type, load_profil): history for model_type, load_profil, history in zip( model_types, load_profiles, histories)}.items())))
        myModels = dict(sorted({(model_type, load_profil): returnedModel for model_type, load_profil, returnedModel in zip( model_types, load_profiles, returnedModels)}.items()))                               

        return train_histories, myModels


if __name__ == "__main__":
    
    test_config = {}
    test_config['models'] = ['xLSTM', 'LSTM', 'Transformer', 'KNN', 'PersistencePrediction']  
    test_config['load_profiles'] = '../data/london_loadprofiles_37households_each.pkl'
    test_config['num_of_communities'] = 10
    test_config['pretraining_mode'] = 'store_and_load_weights'
    myModelTrainer = ModelTrainer(test_config, use_multiprocessing=False)
    train_histories, myModels = myModelTrainer.train()
