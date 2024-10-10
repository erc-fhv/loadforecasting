import multiprocessing as mp
import pandas as pd
import holidays
import pytz
from demandlib import bdew
import pickle

# Imports own modules.
# The imports are done relative to the root of the project.
#
import scripts.Model as model
import scripts.Simulation_config as config
import data.weather_data as weather_data
import scripts.ModelAdapter as ModelAdapter
import scripts.utils as utils


class ModelTrainer:

    def run(self, use_multiprocessing = False):
        
        # Run every single config
        all_train_histories, all_trained_models = {}, {}
        for sim_config in config.configs:
            
            # Fetch and prepare all needed data
            loadprofiles = self.preprocess_data(sim_config)
            
            if use_multiprocessing:
                with mp.Pool() as pool:
                    # Prepare input arguments as a list of tuples and exectue them.
                    tasks = [(model_type, load_profile, sim_config)
                            for model_type in sim_config.UsedModels
                            for load_profile in loadprofiles]
                    results = list(pool.map(self.optimize_model_wrapper, tasks))

            else:   # Single Process
                results = []
                for model_type in sim_config.UsedModels:
                    for load_profile in loadprofiles:
                        result = optimize_model(model_type, load_profile, sim_config)
                        results.append(result)

            # Store the results into dicts
            model_types, load_profiles, sim_configs, histories, returnedModels = zip(*results)
            for i in range(len(model_types)):
                result_key = (model_types[i], load_profiles[i], sim_configs[i])
                all_train_histories[result_key] = histories[i]
                all_trained_models[result_key] = returnedModels[i]
        
        # Persist all results
        utils.Serialize.store_results_with_pickle(all_train_histories)
        utils.Serialize.store_results_with_torch(all_trained_models)
        
        return      

    def preprocess_data(self, sim_config):
        
        print(f"Do Data Preprocessing.", flush=True)
        
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
            out_filename = 'scripts/outputs/file_' + str(i) + '.pkl'
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
            pretraining_filename = 'scripts/outputs/standard_loadprofile.pkl'
            with open(pretraining_filename, 'wb') as file:
                pickle.dump((X, Y, modelAdapter), file)
            
            # Do model pretraining
            for model_type in sim_config.UsedModels:
                print(f"\nPretraining {model_type} model.", flush=True)
                myModel = model.Model(model_type, modelAdapter)
                myModel.train_model(X['train'], Y['train'], pretrain_now=True, finetune_now=False, verbose=0)

        return loadProfiles_filenames              
    
    def optimize_model_wrapper(self, args):
        return optimize_model(*args)

# The following optimization method is defined outside an instance 
# in order to avoid problems with multiprocessing.
# 
def optimize_model(model_type, load_profile, sim_config):
    
    print(f"\nProcessing model {model_type} with load profile {load_profile}", flush=True)

    # Load a new powerprofile
    with open(load_profile, 'rb') as f:
        (X, Y, modelAdapter) = pickle.load(f)

    # Train and evaluate the model
    myModel = model.Model(model_type, modelAdapter=modelAdapter)
    history = myModel.train_model(X['train'], Y['train'], X['test'], Y['test'], 
                                  pretrain_now=False, finetune_now=sim_config.DoTransferLearning, verbose=0)
    history = myModel.evaluate(X['test'], Y['test'], history)
    
    # Return the results
    return (model_type, load_profile, sim_config, history, myModel.my_model)

if __name__ == "__main__":    
    ModelTrainer().run()
