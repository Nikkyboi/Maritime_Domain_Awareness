import sys
from pathlib import Path
from evaluate import evaluate_predictions, print_evaluation_results


def main():
    """
    Main entry point for Kalman Filter baseline evaluation.
    Orchestrates the entire evaluation pipeline.
    """
    # Get project root and data path
    project_root = Path(__file__).resolve().parents[4]
    data_path = str(project_root / "maritime_domain_awareness" / "data" / "Processed" / "MMSI=219001695" / "Segment=0" / "*.parquet")
    
    # Print header
    print("="*60)
    print("KALMAN FILTER BASELINE EVALUATION")
    print("="*60)
    print(f"Data path: {data_path}")
    
    # Run evaluation
    results = evaluate_predictions(
        data_path=data_path,
        prediction_minutes=30,
        train_split=0.8
    )
    
    # Display results
    print_evaluation_results(results)


if __name__ == "__main__":
    src_dir = Path(__file__).resolve().parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    main()