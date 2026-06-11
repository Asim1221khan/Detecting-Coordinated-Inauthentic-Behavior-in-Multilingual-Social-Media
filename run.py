import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.models.cib_detector import MultilingualCIBDetector
from src.utils.logger import get_logger

logger = get_logger("run_entrypoint")

def run_cli(dataset_path: str, output_path: str = None) -> None:
    """Runs the CIB detection pipeline via CLI."""
    logger.info(f"Starting CLI pipeline with dataset: {dataset_path}")
    
    # 1. Load Data
    try:
        raw_df = DataLoader.load_file(dataset_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
        
    # 2. Validate & Clean
    validated_df = DataValidator.clean_nulls(raw_df)
    
    # 3. Preprocess
    pipeline = PreprocessingPipeline()
    processed_df = pipeline.run(validated_df)
    
    # 4. Detect CIB
    detector = MultilingualCIBDetector()
    results_df, _ = detector.fit_predict(processed_df)
    
    # 5. Save Results
    if output_path is None:
        output_dir = Path("reports/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"cib_results_{Path(dataset_path).stem}.csv"
        
    results_df.to_csv(output_path, index=False)
    logger.info(f"CIB analysis successfully finished! Results saved to {output_path}")

def run_streamlit() -> None:
    """Launches the Streamlit dashboard."""
    logger.info("Launching Streamlit dashboard application...")
    # Call streamlit programmatically or via system command
    app_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
    cmd = f"streamlit run \"{app_path}\""
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🛡️ Multilingual Coordinated Inauthentic Behavior (CIB) Detection System"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--streamlit",
        action="store_true",
        help="Start the Streamlit dashboard application."
    )
    group.add_argument(
        "--cli",
        action="store_true",
        help="Execute CIB detection via command line interface."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/samples/sample_twitter.csv",
        help="Path to raw dataset (CSV/TSV/JSON/JSONL) for CLI execution."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output CSV results for CLI execution."
    )
    
    args = parser.parse_args()
    
    if args.streamlit:
        run_streamlit()
    elif args.cli:
        run_cli(args.dataset, args.output)
