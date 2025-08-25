#!/usr/bin/env python3
"""
ET Color Naming Research - Complete Pipeline Runner
Runs all 9 pipeline steps in sequence with proper error handling and logging.
"""

import subprocess
import sys
import logging
import os
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_step(step_num, description, module_path, args=None):
    """Run a single pipeline step."""
    logging.info(f"STEP {step_num}: {description}")
    
    cmd = [sys.executable, "-m", module_path]
    if args:
        cmd.extend(args)
    
    logging.info(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        # Log output if it's not too long
        if result.stdout and len(result.stdout) < 1000:
            logging.info(result.stdout)
        elif result.stdout:
            logging.info(result.stdout[:500] + "... (truncated)")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logging.error(f" Step {step_num} failed after {duration:.2f} seconds:")
        logging.error(f"Return code: {e.returncode}")
        if e.stdout:
            logging.error(f"stdout: {e.stdout}")
        if e.stderr:
            logging.error(f"stderr: {e.stderr}")
        return False, duration
    
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f" Step {step_num} failed with unexpected error after {duration:.2f} seconds: {e}")
        return False, duration

def check_prerequisites():
    """Check if required directories and files exist."""
    
    required_dirs = ['data', 'src', 'spincam']
    required_files = [
        'data/processed_combined_data.csv',
        'data/train_data_with_cam16_ucs.csv', 
        'data/test_data_cam16_ucs.csv'
    ]
    
    missing = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(f"Directory: {dir_path}")
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(f"File: {file_path}")
    
    if missing:
        logging.error(" Missing prerequisites:")
        for item in missing:
            logging.error(f"   - {item}")
        return False
    
    return True

def create_output_directories():
    """Create output directories if they don't exist."""
    dirs = ['results', 'plots', 'models', 'val_results', 'PCw_results', 
            'synonyms_antonyms', 'translation_symmetric']
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def select_steps(steps):
    """Let user select which steps to run."""
    print("\n Available Pipeline Steps:")
    print("="*60)
    
    for step_num, description, _, _ in steps:
        # Check if step outputs already exist
        status = check_step_outputs(step_num)
        print(f"  {step_num}. {description} {status}")
    
    print("\n" + "="*60)
    print("Options:")
    print("  'all' - Run all steps")
    print("  '1,3,5' - Run specific steps (comma-separated)")
    print("  '3-7' - Run range of steps")
    print("  'skip1,skip3' - Run all except specified steps")
    
    while True:
        choice = input("\nEnter your choice: ").strip()
        
        if choice.lower() == 'all':
            return list(range(1, len(steps) + 1))
        
        elif choice.startswith('skip'):
            # Parse skip pattern: skip1,skip3
            skip_nums = []
            skip_part = choice[4:]  # Remove 'skip'
            for item in skip_part.split(','):
                try:
                    skip_nums.append(int(item.strip()))
                except ValueError:
                    continue
            selected = [i for i in range(1, len(steps) + 1) if i not in skip_nums]
            return selected
        
        elif '-' in choice and choice.count('-') == 1:
            # Parse range: 3-7
            try:
                start, end = choice.split('-')
                start, end = int(start.strip()), int(end.strip())
                if 1 <= start <= end <= len(steps):
                    return list(range(start, end + 1))
                else:
                    print(f" Range must be between 1 and {len(steps)}")
            except ValueError:
                print(" Invalid range format. Use format like '3-7'")
        
        else:
            # Parse comma-separated: 1,3,5
            try:
                selected = []
                for item in choice.split(','):
                    num = int(item.strip())
                    if 1 <= num <= len(steps):
                        selected.append(num)
                    else:
                        print(f" Step {num} is out of range (1-{len(steps)})")
                        break
                else:
                    if selected:
                        return sorted(selected)
                    
            except ValueError:
                pass
            
            print(" Invalid input. Try 'all', '1,3,5', '3-7', or 'skip1,skip3'")

def check_step_outputs(step_num):
    """Check if step outputs already exist and return status."""
    status_map = {
        1: ('models/', '*.pkl'),
        2: ('results/', '*_predictions.csv'), 
        3: ('plots/', '*.png'),
        4: ('val_results/', 'BritishEnglish_munsell_results.mat'),
        5: ('', 'max_min_bound outputs'),  # This step prints to console
        6: ('PCw_results/', 'PCw_*.csv'),
        7: ('synonyms_antonyms/', '*.csv'),
        8: ('translation_symmetric/', '*.csv'),
        9: ('results/', '*_interface.csv')
    }
    
    if step_num not in status_map:
        return ""
    
    dir_path, pattern = status_map[step_num]
    
    if step_num == 5:  # Special case for max_min_bound
        return "(prints to console)"
    
    if not os.path.exists(dir_path):
        return " (outputs missing)"
    
    files = os.listdir(dir_path)
    if step_num == 1:  # Models
        model_files = [f for f in files if f.endswith('.pkl')]
        if len(model_files) >= 5:  # Expecting 5 language models
            return " (models exist)"
    elif step_num == 2:  # Predictions
        pred_files = [f for f in files if f.endswith('_predictions.csv')]
        if len(pred_files) >= 5:  # Expecting 5 language predictions
            return " (predictions exist)"
    elif step_num == 3:  # Plots
        plot_files = [f for f in files if f.endswith('.png')]
        if len(plot_files) >= 10:  # Expecting ~10 plot files
            return " (plots exist)"
    elif step_num == 4:  # Munsell
        if 'BritishEnglish_munsell_results.mat' in files:
            return " (results exist)"
    elif step_num == 6:  # PCw results
        pcw_files = [f for f in files if f.startswith('PCw_') and f.endswith('.csv')]
        if len(pcw_files) >= 5:
            return " (PCw files exist)"
    elif step_num == 7:  # Synonyms/Antonyms
        if files:
            return " (analysis exists)"
    elif step_num == 8:  # Translations
        if files:
            return " (translations exist)"
    elif step_num == 9:  # Interface
        interface_files = [f for f in files if f.endswith('_interface.csv')]
        if len(interface_files) >= 5:
            return " (interface files exist)"
    
    return " (outputs missing)"

def main():
    """Run the complete pipeline."""
    start_time = datetime.now()
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                   Color Naming Pipeline                   ║
    ║                          Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}                          ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    logging.info(f" Starting ET Color Naming Pipeline")
    
    # Check prerequisites
    if not check_prerequisites():
        logging.error("Prerequisites check failed. Exiting.")
        sys.exit(1)
    
    # Create output directories
    create_output_directories()
    
    # Define all pipeline steps
    steps = [
    (1, "Evaluating Multi-language Models", "src.evaluation.opt_eval", []),
    (2, "Predicting Multi-language Colour Names for J slices", "src.prediction.get_predictions_for_Jslices", []),
    (3, "Generating J slices and Frequnecy Plots ", "src.visualization.viz", []),
    (4, "Munsell Colour Analysis (British English)", "src.analysis.munsell", []),
    (5, "Munsell Visualization", "src.visualization.munsell_viz",[]), 
    (6, "Running Baseline Models", "src.analysis.max_min_bound", []),           
    (7, "Probability Matrix Processing", "src.analysis.get_redcuedPCw", []),    
    (8, "Synonyms & Antonyms Analysis", "src.analysis.get_synonyms_and_antonyms", []),  
    (9, "Cross-language Translation Analysis", "src.analysis.get_translations", ["--all-pairs"]), 
    (10, "Interface Data Generation", "src.prediction.get_interface_data", []), 
    ]
    
    # Let user select steps to run
    selected_step_numbers = select_steps(steps)
    selected_steps = [step for step in steps if step[0] in selected_step_numbers]
    
    print(f"\n Running {len(selected_steps)} steps: {[s[0] for s in selected_steps]}")
    
    if len(selected_steps) == 0:
        print("No steps selected. Exiting.")
        return
    
    # Run selected steps
    successful_steps = 0
    failed_steps = []
    step_durations = []
    
    for step_info in selected_steps:
        step_num = step_info[0]
        description = step_info[1]
        module_path = step_info[2]
        args = step_info[3]
        
        success, duration = run_step(step_num, description, module_path, args)
        step_durations.append(duration)
        
        if success:
            successful_steps += 1
        else:
            failed_steps.append((step_num, description))
            
            # Ask user if they want to continue
            response = input(f"\n Step {step_num} failed. Continue with remaining steps? (y/n): ")
            if response.lower() != 'y':
                logging.info("Pipeline stopped by user.")
                break
    
    # Final summary
    end_time = datetime.now()
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                        PIPELINE COMPLETE                              ║
    ║                        Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}                         ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    
    

if __name__ == "__main__":
    main()
