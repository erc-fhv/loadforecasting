# Referenzes: 
#   - Code and Tutorial:
#       https://github.com/meteostat/meteostat-python
#   - API Documentation:
#       https://dev.meteostat.net/python/hourly.html#data-structure
#   - Weather station:
#       https://meteostat.net/de/place/gb/london?s=03772&t=2012-03-14/2012-03-14
#

from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly

class WeatherMeasurements:

    def get_data(self, startDate, endDate, lat, lon, alt, sample_periode, tz):

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
