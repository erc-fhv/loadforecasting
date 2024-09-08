import pandas as pd

class StandardProfiles_Readout:
    
    def __init__(
                self,
                profiles_path = '../../data/raw/',
                profiles_folder = 'CSV_74_Loadprofiles_1min_W_var/',
                ):        
        
        self.profiles_path = profiles_path
        self.profiles_folder = profiles_folder

    def get_raw_standardprofiles(
                self, 
                profilesType = 1,   # type 1 = households
                tz = 'Europe/Vienna',
                ):

        # Read in the Standard-Power-Profiles CSV file
        df = pd.read_csv('../../data/raw/Standard_Loadprofile_APCS_2010.csv', delimiter=';')

        # Combine the 'Date' and 'Time' columns and convert to datetime format
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d.%m.%Y %H:%M:%S')

        # Set the combined datetime column as the index
        df.set_index('Datetime', inplace=True)

        # Drop the 'Date' and 'Time' columns
        df.drop(['Date', 'Time'], axis=1, inplace=True)

        # Filter rows where 'Type' column is equal to profilesType
        standardprofiles = df[df['Type'] == profilesType]

        # Set timezone
        standardprofiles = standardprofiles.tz_localize('UTC').tz_convert(tz)

        return standardprofiles

    def get_scaled_standardprofiles(self, powerProfile):

        # Scale the standard load profile to [W].
        # Multiplicator, to come from [kWh] per 15min to [W]
        # = 10000 * real_consumption_kWh / (1000kWh * (60min/15min))
        #
        powerProfile_1h = powerProfile.copy().resample('1h').mean()  # Average power consumption in [W] within 1h of all households.
        total_characteristic_energy_consumption = powerProfile_1h.sum()    # Total energy consumption in [Wh] consumed by all households within one year.
        standardprofiles = self.get_raw_standardprofiles()
        total_standard_energy_consumption = standardprofiles.Value.sum()     # Total energy consumption in [kWh] within one year.
        standard_profile_factor = total_characteristic_energy_consumption / total_standard_energy_consumption * 4.0
        standardprofiles_scaled = standardprofiles.Value * standard_profile_factor
        
        return standardprofiles_scaled


if __name__ == "__main__":
    pass
