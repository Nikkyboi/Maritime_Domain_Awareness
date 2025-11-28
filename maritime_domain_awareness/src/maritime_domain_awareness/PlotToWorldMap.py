import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import matplotlib.pyplot as plt
import numpy as np

def PlotToWorldMap(train_val_data=None, test_actual=None, test_predicted=None):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.add_feature(NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='lightgreen'))
    ax.add_feature(NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='lightblue'))

   
    def extract_coords(data):
        data_np = data.detach().cpu().numpy()
        return data_np[:, 0], data_np[:, 1] 

    lats, lons = extract_coords(train_val_data)
    ax.plot(lons, lats, '-', color='blue', label='Train+Val', transform=ccrs.PlateCarree())

    lats, lons = extract_coords(test_actual)
    ax.plot(lons, lats, 'o-', color='green', label='Actual', transform=ccrs.PlateCarree())

    lats, lons = extract_coords(test_predicted)
    ax.plot(lons, lats, 'x-', color='red', label='Predicted', transform=ccrs.PlateCarree())

    ax.legend()
    plt.show()