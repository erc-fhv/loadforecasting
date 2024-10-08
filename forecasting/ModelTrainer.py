from tqdm import tqdm
import pickle
import multiprocessing as mp
import pandas as pd
import holidays
import pytz
from demandlib import bdew
import pickle
from datetime import datetime

# Imports own modules.
# The imports are done relative to the root of the project.
#
import forecasting.Model as model
import forecasting.Simulation_config as config
import data.weather_data as weather_data
import forecasting.ModelAdapter as ModelAdapter


class ModelTrainer:

    def __init__(self, use_multiprocessing = False):        
        self.use_multiprocessing = use_multiprocessing

    def run(self):
        
        # Run every single config
        all_train_histories, all_trained_models = [], []
        for sim_config in config.configs:
            
            # Fetch and prepare all needed data
            loadprofiles = self.preprocess_data(sim_config)
            
            if self.use_multiprocessing:
                with mp.Pool() as pool:
                    # Prepare input arguments as a list of tuples and exectue them.
                    tasks = [(model_type, load_profile, sim_config)
                            for model_type in sim_config.UsedModels
                            for load_profile in loadprofiles]
                    results = list(
                        tqdm(
                            pool.map(self.optimize_model_wrapper, tasks),
                            total=len(tasks)
                        )
                    )

            else:   # Single Process
                results = []
                for model_type in tqdm(sim_config.UsedModels, position=0):
                    for load_profile in tqdm(loadprofiles, position=1, leave=False):
                        result = optimize_model(model_type, load_profile, sim_config)
                        results.append(result)

            # Create a list of dictionaries out of the results
            model_types, load_profiles, sim_configs, histories, returnedModels = zip(*results)
            train_histories_unsorted = {(model_type, load_profil, sim_config): history for model_type, load_profil, sim_config, history in zip(model_types, load_profiles, sim_configs, histories)}
            train_histories = dict(sorted((train_histories_unsorted.items())))
            myModels_unsorted = {(model_type, load_profil, sim_config): returnedModel for model_type, load_profil, sim_config, returnedModel in zip(model_types, load_profiles, sim_configs, returnedModels)}
            myModels = dict(sorted(myModels_unsorted.items()))                               
            all_train_histories.append(train_histories)
            all_trained_models.append(myModels)
        
        # Pickle all results
        self.store_results(all_train_histories, all_trained_models)
        
        return

    def preprocess_data(self, sim_config):
        
        # Readout the power profiles, bring them to the format needed by the model and store those profiles
        #
        loadProfiles = pd.read_pickle(sim_config.Aggregation_Count)
        loadProfiles = loadProfiles[:sim_config.NrOfComunities]

        # Readout the weather data
        #
        startDate = loadProfiles[0].index[0].to_pydatetime().replace(tzinfo=None)
        endDate = loadProfiles[0].index[-1].to_pydatetime().replace(tzinfo=None)
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

        # Bring the power profiles to the model shape of (nr_of_batches, timesteps, features)
        #
        loadProfiles_filenames = []
        for i, powerProfile in enumerate(loadProfiles[:sim_config.NrOfComunities]):
            
            # Preprocess data to get X and Y for the model
            out_filename = 'forecasting/outputs/file_' + str(i) + '.pkl'
            modelAdapter = ModelAdapter.ModelAdapter(public_holidays_timestamps, train_size = 466,
                                                    addLaggedPower=True, shuffle_data=False)

            X, Y = modelAdapter.transformData(powerProfile, weatherData)
            with open(out_filename, 'wb') as file:
                pickle.dump((X, Y, modelAdapter), file)   
            loadProfiles_filenames.append(out_filename)
        
        # If required, do pretraing
        if sim_config.DoPretraining:

            # Load the BDEW standard load profiles for the desired datetime range
            standard_loadprofiles = []
            for year in range(startDate.year, endDate.year + 1):
                load_profile = bdew.ElecSlp(year, holidays=public_holidays_timestamps).get_profile({"h0": 1000})
                standard_loadprofiles.append(load_profile)
            all_standard_loadprofiles = pd.concat(standard_loadprofiles)['h0']
            all_standard_loadprofiles = all_standard_loadprofiles[(all_standard_loadprofiles.index >= startDate) & (all_standard_loadprofiles.index <= endDate)]
            
            # Preprocess data to get X and Y for the model
            modelAdapter = ModelAdapter.ModelAdapter(public_holidays_timestamps, train_size = len(all_standard_loadprofiles), 
                                                    dev_size = 0, addLaggedPower=True, shuffle_data=False)
            X, Y = modelAdapter.transformData(all_standard_loadprofiles, weatherData=None)
            pretraining_filename = 'forecasting/outputs/standard_loadprofile.pkl'
            with open(pretraining_filename, 'wb') as file:
                pickle.dump((X, Y, modelAdapter), file)
            
            # Do model pretraining
            for model_type in tqdm(sim_config.UsedModels):
                num_of_features = X['train'].shape[2]
                myModel = model.Model(num_of_features, model_type, modelAdapter)
                myModel.train_model(X['train'], Y['train'], pretrain_now=True, finetune_now=False, verbose=0)

        return loadProfiles_filenames
    
    def store_results(self, all_train_histories, all_trained_models):      
        
        # Store both variables in a pickle files with the timestamp
        timestamp = datetime.now().strftime("_%Y%m%d_%H%M")
        filename = f"forecasting/outputs/model_evaluation{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump((all_train_histories, all_trained_models), f)
        print(f"Results stored in: {filename}")
    
    def optimize_model_wrapper(self, args):
        return optimize_model(*args)

# The following optimization method is defined outside an instance 
# in order to avoid problems with multiprocessing.
# 
def optimize_model(model_type, load_profile, sim_config):

    # Load a new powerprofile
    with open(load_profile, 'rb') as f:
        (X, Y, modelAdapter) = pickle.load(f)

    # Train and evaluate the model
    num_of_features = X['train'].shape[2]
    myModel = model.Model(num_of_features, model_type, modelAdapter=modelAdapter)
    history = myModel.train_model(X['train'], Y['train'], X['test'], Y['test'], 
                                  pretrain_now=False, finetune_now=sim_config.DoTransferLearning, verbose=0)
    history = myModel.evaluate(X['test'], Y['test'], history)
    
    # Return the results
    return (model_type, load_profile, sim_config, history, myModel)

if __name__ == "__main__":    
    ModelTrainer(use_multiprocessing=False).run()

