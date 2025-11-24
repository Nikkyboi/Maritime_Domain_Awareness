import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import pyarrow.dataset as ds

# Add the current directory to sys.path to allow importing preproccesing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from preproccesing import interpolate, filter_anomalies
except ImportError:
    # Fallback if running from root
    from maritime_domain_awareness.src.maritime_domain_awareness.preproccesing import interpolate, filter_anomalies

def compare_paths(mmsi, raw_path):
    """
    Visualizes the difference between raw and interpolated paths for a given MMSI.
    """
    print(f"Loading data for MMSI: {mmsi} from {raw_path}...")
    
    try:
        # Use pyarrow.dataset for more robust reading of partitioned datasets
        dataset = ds.dataset(raw_path, format="parquet", partitioning="hive")
        
        # Try filtering as string first
        try:
            table = dataset.to_table(filter=(ds.field("MMSI") == str(mmsi)))
            df_raw = table.to_pandas()
        except Exception:
             # If that fails, try as int
            table = dataset.to_table(filter=(ds.field("MMSI") == int(mmsi)))
            df_raw = table.to_pandas()
            
    except Exception as e:
        print(f"Error reading data: {e}")
        print("Tip: If you see 'Repetition level histogram size mismatch', try deleting the 'raw_segmented.parquet' folder and running preprocessing again.")
        return

    if df_raw.empty:
        print(f"No data found for MMSI {mmsi}")
        return

    segments = df_raw['Segment'].unique()
    num_segments = len(segments)
    print(f"Found {num_segments} segments for MMSI {mmsi}")

    if num_segments == 0:
        return

    # Determine grid size for subplots
    # We'll limit to max 9 plots (3x3) to keep it readable
    max_plots = 9
    if num_segments > max_plots:
        print(f"Showing first {max_plots} segments out of {num_segments}...")
        plot_segments = segments[:max_plots]
    else:
        plot_segments = segments

    import math
    cols = min(3, len(plot_segments))
    rows = math.ceil(len(plot_segments) / cols)
    
    # --- PLOT 1: RAW DATA ---
    print("Plotting Raw Data (Close window to see Interpolated Data)...")
    fig_raw, axes_raw = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
    
    # Ensure axes is always a list/array even if only 1 plot
    if len(plot_segments) == 1:
        axes_raw = [axes_raw]
    else:
        axes_raw = axes_raw.flatten()

    for i, seg in enumerate(plot_segments):
        ax = axes_raw[i]
        df_seg = df_raw[df_raw['Segment'] == seg].copy()
        
        if len(df_seg) < 2:
            ax.text(0.5, 0.5, "Too few points", ha='center', va='center')
            continue

        # Plot raw points
        ax.scatter(df_seg['Longitude'], df_seg['Latitude'], 
                   c='red', s=10, marker='x', label='Raw Pts', alpha=0.7)
        
        # Add start/end markers
        ax.scatter(df_seg['Longitude'].iloc[0], df_seg['Latitude'].iloc[0], 
                   c='green', s=50, marker='^', label='Start', zorder=10)
        ax.scatter(df_seg['Longitude'].iloc[-1], df_seg['Latitude'].iloc[-1], 
                   c='black', s=50, marker='v', label='End', zorder=10)

        ax.set_title(f"Segment {seg} (RAW)")
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        if i == 0:
            ax.legend(fontsize='small')

    # Hide unused subplots
    for j in range(len(plot_segments), len(axes_raw)):
        axes_raw[j].axis('off')

    fig_raw.suptitle(f"RAW DATA - MMSI: {mmsi}", fontsize=16)
    plt.show()

    # --- PLOT 2: INTERPOLATED DATA ---
    print("Plotting Interpolated Data...")
    fig_int, axes_int = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
    
    if len(plot_segments) == 1:
        axes_int = [axes_int]
    else:
        axes_int = axes_int.flatten()

    for i, seg in enumerate(plot_segments):
        print(f"Processing Segment {seg}...")
        ax = axes_int[i]
        df_seg = df_raw[df_raw['Segment'] == seg].copy()
        
        if len(df_seg) < 2:
            ax.text(0.5, 0.5, "Too few points", ha='center', va='center')
            continue

        # Filter anomalies before interpolation
        df_clean = filter_anomalies(df_seg)
        
        if df_clean.empty:
             ax.text(0.5, 0.5, "Filtered as anomaly", ha='center', va='center')
             continue

        # Apply interpolation
        df_interp = interpolate(df_clean)
        
        # Calculate time info
        start_time = df_seg['Timestamp'].min()
        end_time = df_seg['Timestamp'].max()
        duration = end_time - start_time
        
        # Create time-based coloring for interpolated points
        times = (df_interp['Timestamp'] - df_interp['Timestamp'].min()).dt.total_seconds()
        
        # Plot interpolated path (line) - gray for background
        ax.plot(df_interp['Longitude'], df_interp['Latitude'], 
                'k-', label='Path Line', alpha=0.2, linewidth=0.5)
        
        # Plot interpolated points colored by time
        sc = ax.scatter(df_interp['Longitude'], df_interp['Latitude'], 
                   c=times, cmap='viridis', s=3, alpha=0.8, label='Interp Pts (Time)')
        
        # Add start/end markers
        ax.scatter(df_seg['Longitude'].iloc[0], df_seg['Latitude'].iloc[0], 
                   c='green', s=50, marker='^', label='Start', zorder=10)
        ax.scatter(df_seg['Longitude'].iloc[-1], df_seg['Latitude'].iloc[-1], 
                   c='black', s=50, marker='v', label='End', zorder=10)

        ax.set_title(f"Segment {seg} (INTERPOLATED)\n{duration}", fontsize=10)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Only add legend to the first plot to save space
        if i == 0:
            ax.legend(fontsize='small')
            # Add colorbar to show time progression
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Time (seconds from start)', fontsize=8)

    # Hide unused subplots
    for j in range(len(plot_segments), len(axes_int)):
        axes_int[j].axis('off')

    fig_int.suptitle(f"INTERPOLATED DATA - MMSI: {mmsi}", fontsize=16)
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    # Robust path calculation
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Walk up until we find the 'data' directory
    current_dir = script_dir
    project_root = None
    while True:
        potential_data_dir = os.path.join(current_dir, "data")
        if os.path.exists(potential_data_dir):
            project_root = current_dir
            break
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir: # Hit root
            break
        current_dir = parent_dir
    
    if project_root:
        RAW_DATA_PATH = os.path.join(project_root, "data/preprocessed/raw_segmented.parquet")
    else:
        # Fallback if auto-discovery fails (e.g. structure is different)
        # Assuming standard structure: root/maritime_domain_awareness/src/maritime_domain_awareness/compare.py
        # We want root/maritime_domain_awareness/data/...
        # So we go up 3 levels from script_dir
        project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
        RAW_DATA_PATH = os.path.join(project_root, "maritime_domain_awareness/data/preprocessed/raw_segmented.parquet")

    print(f"Looking for data at: {RAW_DATA_PATH}")

    if len(sys.argv) > 1:
        mmsi_input = sys.argv[1]
    else:
        mmsi_input = input("Enter MMSI to visualize: ")
        
    compare_paths(mmsi_input, RAW_DATA_PATH)
