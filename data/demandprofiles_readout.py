import pandas as pd

class DemandProfiles_Readout:
    
    def __init__(
                self,
                profiles_path = '../../data/raw/',
                profiles_folder = 'CSV_74_Loadprofiles_1min_W_var/',
                ):
        self.profiles_path = profiles_path
        self.profiles_folder = profiles_folder

    def create_powerconsumption_dataframe(
            self, 
            power_line_nr, 
            calc_aggregated_power=True,
            tz = 'Europe/Vienna',
            ):

        # Read the given datetimes into a DataFrame and convert it into a single datetime column
        #
        datetimes = pd.read_csv(self.profiles_path + self.profiles_folder + 'time_datevec_MEZ.csv', header=None)
        given_column_names = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']
        datetimes.columns = given_column_names
        datetimes['Timestamp'] = pd.to_datetime(datetimes)
        datetimes = datetimes.drop(labels=given_column_names, axis=1)

        # Read in and setup the given power time series
        #
        file_path = self.profiles_path  + self.profiles_folder+ 'PL' + str(power_line_nr) + '.csv'
        values = pd.read_csv(file_path, header=None)
        data = pd.concat([datetimes, values], axis=1)
        data.set_index('Timestamp', inplace=True)

        if calc_aggregated_power == True:
            # Aggregate all households and add it to the dataframe
            #
            # Add a new column to the DataFrame with the row sums
            data['Aggregated_Power'] = data.sum(axis=1)

        # Set timezone
        data.index = data.index.tz_localize('UTC+01:00').tz_convert(tz)

        return data

    def get_aggregated_powerprofile(self):

        # Readout the characteristic demand profiles
        #
        data_PL1 = self.create_powerconsumption_dataframe(1)
        data_PL2 = self.create_powerconsumption_dataframe(2)
        data_PL3 = self.create_powerconsumption_dataframe(3)

        # Prepare the power profile:
        # Aggregate all households and phases
        #
        powerProfile =  data_PL1['Aggregated_Power'] + \
                        data_PL2['Aggregated_Power'] + \
                        data_PL3['Aggregated_Power']
        return powerProfile

if __name__ == "__main__":
    pass
