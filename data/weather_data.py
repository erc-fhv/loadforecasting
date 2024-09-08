# Referenzes: 
#   - Code and Tutorial:
#       https://github.com/meteostat/meteostat-python
#   - API Documentation:
#       https://dev.meteostat.net/python/hourly.html#data-structure
#   - Weather station:
#       https://meteostat.net/de/place/de/bochum?s=EUMPC&t=2010-01-01/2010-01-01
#       https://meteostat.net/de/place/gb/london?s=EGLC0&t=2012-03-14/2012-03-14
#

from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly

class WeatherMeasurements:
    
    def __init__(self):
        pass

    def get_data(
            self,
            startDate = datetime(2010, 1, 1, 0, 0), 
            endDate = datetime(2010, 12, 31, 23, 55),
            lat = 51.4817,      # Default Location:
            lon = 7.2165,       # Bochum Germany,
            alt = 102,          # Meteostat weatherstation   
            sample_periode = 'hourly', 
            tz = 'Europe/Vienna',
            ):

        # Create Geo-Point
        location = Point(lat, lon, alt)

        # Specify sampling periode
        if sample_periode == 'hourly':
            self.data = Hourly(location, startDate, endDate)
        elif sample_periode == 'daily':
            self.data = Daily(location, startDate, endDate)
        else:
            raise ValueError("Invalid sample_periode chosen.")

        # Download selected data
        self.data = self.data.fetch()

        # Replace NaN values with zero (if there are any)
        self.data.fillna(0, inplace=True)

        # Set timezone
        self.data.index = self.data.index.tz_localize('UTC').tz_convert(tz)

        return self.data

    def plot_data(self):
        # Plot line chart
        data = self.get_data()
        data.plot(y=['tsun'])
        data.plot(y=['tavg', 'prcp', 'wspd'])
        plt.show()

if __name__ == "__main__":
    weather_measurements = WeatherMeasurements()
    weather_measurements.plot_data()


