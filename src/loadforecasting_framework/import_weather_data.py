# Referenzes:
#   - Code and Tutorial:
#       https://github.com/meteostat/meteostat-python
#   - API Documentation:
#       https://dev.meteostat.net/python/hourly.html#data-structure
#   - Weather station:
#       https://meteostat.net/de/place/gb/london?s=03772&t=2012-03-14/2012-03-14
#

from meteostat import Point, Daily, Hourly

class WeatherMeasurements:

    def get_data(self, startDate, endDate, lat, lon, alt, sample_periode, tz):

        # Create Geo-Point
        location = Point(lat, lon, alt)

        # Specify sampling periode
        if sample_periode == 'hourly':
            # Fix deprecated warning "Use of '1H' frequency is deprecated, use '1h' instead."
            Hourly._freq = "1h"
            self.data = Hourly(location, startDate, endDate, tz)
        elif sample_periode == 'daily':
            self.data = Daily(location, startDate, endDate, tz)
        else:
            raise ValueError("Invalid sample_periode chosen.")

        # Download selected data
        self.data = self.data.fetch()

        # Replace NaN values with zero (if there are any)
        self.data.fillna(0, inplace=True)

        # Check the timezone
        assert str(self.data.index.tz) == str(tz), f"Expected tz = {tz}, \
            received tz = {self.data.index.tz}"

        return self.data
