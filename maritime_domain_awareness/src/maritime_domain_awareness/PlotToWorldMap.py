import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import matplotlib.pyplot as plt

def PlotToWorldMap(actualPoint=None, predictedPoint=None):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.add_feature(NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='lightgreen'))
    ax.add_feature(NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='lightblue'))

    # Flatten and convert tensors to lists of coordinates
    actual_lons = []
    actual_lats = []
    predicted_lons = []
    predicted_lats = []

    for act, pred in zip(actualPoint, predictedPoint):
        act_np = act.detach().cpu().numpy()  # shape: (batch, seq_len, 2) or (seq_len, 2)
        pred_np = pred.detach().cpu().numpy()
        # Flatten all points
        act_np = act_np.reshape(-1, 2)
        pred_np = pred_np.reshape(-1, 2)
        for pt in act_np:
            actual_lons.append(pt[0])
            actual_lats.append(pt[1])
        for pt in pred_np:
            predicted_lons.append(pt[0])
            predicted_lats.append(pt[1])

    ax.plot(actual_lons, actual_lats, 'o-', color='green', label='Actual', transform=ccrs.PlateCarree())
    ax.plot(predicted_lons, predicted_lats, 'x-', color='red', label='Predicted', transform=ccrs.PlateCarree())
    ax.legend()
    plt.show()