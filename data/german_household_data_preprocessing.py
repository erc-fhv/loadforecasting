import pandas as pd
import numpy as np
import pathlib
import sys
from copy import copy

sys.path.append('../../')

class LoadData:
    raw_data_dir          = pathlib.Path('data/raw')
    preprocessed_data_dir = pathlib.Path('data/preprocessed')

    def get_preprocessed_data(self, name):
        path =  self.preprocessed_data_dir.joinpath(name+'.pkl')
        if pathlib.Path.exists(path):
            return pd.read_pickle(path)
        return pd.DataFrame([])

    def preprocess_data(self, num, name):
        print('preprocessing load data...')
        # Load Data
        pl1 = pd.read_csv(self.raw_data_dir.joinpath('CSV_74_Loadprofiles_1min_W_var/PL1.csv'), header=None) # Phase 1
        pl2 = pd.read_csv(self.raw_data_dir.joinpath('CSV_74_Loadprofiles_1min_W_var/PL2.csv'), header=None) # Phase 2
        pl3 = pd.read_csv(self.raw_data_dir.joinpath('CSV_74_Loadprofiles_1min_W_var/PL3.csv'), header=None) # Phase 3
        # Load Index
        datetimeindex = pd.read_csv(
            self.raw_data_dir.joinpath('CSV_74_Loadprofiles_1min_W_var/time_datevec_MEZ.csv'), 
            header=None, 
            parse_dates={'datetime': [0, 1, 2, 3, 4, 5]}, 
            date_format="%Y %m %d %H %M %S")\
                .set_index('datetime')\
                .tz_localize('UTC+01:00')\
                .tz_convert('Europe/Vienna').index # Timestamp of data is local winter time (MEZ = UTC+01:00) -> convert to Local time zone 

        # Preprocess Data
        household_loads = -pl1-pl2-pl3  # Sum Phases. Negative loads for consumtion!!!!!
        household_loads.index = datetimeindex # Assign Index

        household_loads = household_loads.resample('h').mean()

        # repeat december and january data
        december = household_loads['2010-11-01':]
        december.index = december.index - pd.DateOffset(years=1)
        january = household_loads[:'2010-01-31']
        january.index = january.index + pd.DateOffset(years=1)

        household_loads = pd.concat([december, household_loads, january])
        # household_loads = household_loads.loc[]

        #select num profiles out of the list
        assert len(household_loads.columns) >= num, 'The requested number of households can not be provided, as there are not enough profiles in the data'
        
        # np.random.seed(0) # fixe seed for reproducibility
        # cols_sim = list(np.random.choice(household_loads.columns, num, replace=False)) # not using random choice anymore to load the same profiles any time
        cols_sim = list(np.arange(num))
        cols_not_sim = [i for i in household_loads.columns if i not in cols_sim]

        household_loads_sim = household_loads.loc[:, cols_sim] 
        household_loads_not_sim  = household_loads.loc[:, cols_not_sim]

        # Implement param save as in PV Model
        # self.original_load_profile_names = self.data.columns # safe original columns for analysis   
        # self.data.columns = np.arange(len(self.original_load_profile_names))  # rename columns for easier handling

        # Save data
        household_loads_sim.to_pickle(self.preprocessed_data_dir.joinpath(name+'.pkl'))
        household_loads_not_sim.to_pickle(self.preprocessed_data_dir.joinpath(name+'_not_used_for_sim.pkl'))