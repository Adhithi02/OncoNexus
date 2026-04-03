import os
import argparse
from src.utils import set_seed, ensure_directories_exist, setup_logger
from src.phase1_data.phase1 import run_phase1
from src.phase2_network.phase2 import run_phase2
from src.phase3_train.phase3 import run_phase3
from src.phase4_perturbation.phase4 import run_phase4
from src.phase5_biomarkers.phase5 import run_phase5

def main():
    parser = argparse.ArgumentParser(description="OncoNexus Backend ML Pipeline")
    parser.add_argument('--phase', type=str, default="all", help="Phase to run: 1, 2, 3, 4, 5 or 'all'")
    args = parser.parse_args()

    # Apply constraints
    set_seed(42)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    logs_dir = os.path.join(base_dir, "logs")

    ensure_directories_exist([data_dir, models_dir, logs_dir])

    logger = setup_logger("Main")
    logger.info(f"Executing phase(s): {args.phase}")

    phases = str(args.phase).lower()

    try:
        if phases in ["1", "all"]:
            run_phase1(data_dir)
        if phases in ["2", "all"]:
            run_phase2(data_dir)
        if phases in ["3", "all"]:
            run_phase3(data_dir, models_dir, logs_dir)
        if phases in ["4", "all"]:
            run_phase4(data_dir, models_dir)
        if phases in ["5", "all"]:
            run_phase5(data_dir)
            
    except Exception as e:
        logger.error(f"Pipeline failed at phase {phases}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
