import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
import matplotlib.pyplot as plt
import numpy as np
import torch

def PlotToWorldMap(actualPoint=None, model_predictions=None, model_names=None):
    """
    Plot actual trajectory vs multiple model predictions on a world map.
    
    Args:
        actualPoint: Actual trajectory points. Can be:
            - torch.Tensor with shape (N, 2) where columns are [lat, lon]
            - list/array with shape (N, 2)
        model_predictions: Dictionary or list of model predictions
            - If dict: {model_name: predictions} where predictions is (N, 2) [lat, lon]
            - If list: list of predictions, each (N, 2) [lat, lon]
        model_names: List of model names (only used if model_predictions is a list)
    """
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.add_feature(NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='lightgreen'))
    ax.add_feature(NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor='lightblue'))

    # Convert actual points to numpy array [lat, lon]
    if actualPoint is not None:
        if isinstance(actualPoint, torch.Tensor):
            actual_np = actualPoint.detach().cpu().numpy()
        else:
            actual_np = np.array(actualPoint)

        # Ensure shape is (N, 2) [lat, lon]
        if actual_np.ndim == 1:
            actual_np = actual_np.reshape(-1, 2)

        actual_lats = actual_np[:, 0]
        actual_lons = actual_np[:, 1]

        # Plot actual trajectory
        ax.plot(actual_lons, actual_lats, 'o-', color='blue', linewidth=2, 
                markersize=4, label='Actual Trajectory', transform=ccrs.PlateCarree(), zorder=10)

    # Define colors for different models
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
    markers = ['x', '^', 's', 'D', 'v', '<', '>', 'p']

    # Plot model predictions
    if model_predictions is not None:
        if isinstance(model_predictions, dict):
            for idx, (model_name, predictions) in enumerate(model_predictions.items()):
                if isinstance(predictions, torch.Tensor):
                    pred_np = predictions.detach().cpu().numpy()
                else:
                    pred_np = np.array(predictions)

                # Ensure shape is (N, 2) [lat, lon]
                if pred_np.ndim == 1:
                    pred_np = pred_np.reshape(-1, 2)

                pred_lats = pred_np[:, 0]
                pred_lons = pred_np[:, 1]

                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]

                ax.plot(pred_lons, pred_lats, marker=marker, linestyle='--', 
                       color=color, linewidth=1.5, markersize=5,
                       label=f'{model_name} Prediction', 
                       transform=ccrs.PlateCarree(), alpha=0.7)

        elif isinstance(model_predictions, list):
            if model_names is None:
                model_names = [f'Model {i+1}' for i in range(len(model_predictions))]

            for idx, predictions in enumerate(model_predictions):
                if isinstance(predictions, torch.Tensor):
                    pred_np = predictions.detach().cpu().numpy()
                else:
                    pred_np = np.array(predictions)

                # Ensure shape is (N, 2) [lat, lon]
                if pred_np.ndim == 1:
                    pred_np = pred_np.reshape(-1, 2)

                pred_lats = pred_np[:, 0]
                pred_lons = pred_np[:, 1]

                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]

                ax.plot(pred_lons, pred_lats, marker=marker, linestyle='--', 
                       color=color, linewidth=1.5, markersize=5,
                       label=f'{model_names[idx]} Prediction', 
                       transform=ccrs.PlateCarree(), alpha=0.7)

    # Set extent to show Denmark (wider view)
    if actualPoint is not None:
        # Use fixed extent that covers Denmark coastline
        # Denmark roughly: lat 54.5-58°N, lon 8-13°E
        lat_center = (actual_lats.max() + actual_lats.min()) / 2
        lon_center = (actual_lons.max() + actual_lons.min()) / 2

        # Zoom out to show ~3 degrees in each direction (covers Denmark)
        ax.set_extent([
            lon_center - 3,
            lon_center + 3,
            lat_center - 2,
            lat_center + 2
        ], crs=ccrs.PlateCarree())

    ax.legend(loc='upper left')
    ax.gridlines(draw_labels=True, alpha=0.3)
    plt.title('Maritime Trajectory Prediction: Actual vs Model Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig, ax