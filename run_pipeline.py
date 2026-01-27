"""
SCF Color Naming Pipeline Runner
Orchestrates execution of SCF color naming analysis pipeline steps.
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
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        
        if result.stdout:
            log_msg = result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout
            logging.info(log_msg)
        return True, duration
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logging.error(f"Step {step_num} failed after {duration:.2f}s (code: {e.returncode})")
        if e.stderr:
            logging.error(f"Error: {e.stderr}")
        return False, duration
    
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Step {step_num} failed: {e}")
        return False, duration

def create_directories():
    """Create required directories if they don't exist."""
    for dir_name in ['results', 'plots', 'models', 'data']:
        os.makedirs(dir_name, exist_ok=True)

def parse_step_selection(choice, max_steps):
    """Parse user's step selection choice."""
    choice = choice.strip().lower()
    
    if choice == 'all':
        return list(range(1, max_steps + 1))
    
    if choice.startswith('skip'):
        skip_nums = [int(x.strip()) for x in choice[4:].split(',') if x.strip().isdigit()]
        return [i for i in range(1, max_steps + 1) if i not in skip_nums]
    
    if '-' in choice and choice.count('-') == 1:
        try:
            start, end = map(int, choice.split('-'))
            if 1 <= start <= end <= max_steps:
                return list(range(start, end + 1))
        except ValueError:
            pass
    
    try:
        selected = [int(x.strip()) for x in choice.split(',')]
        if all(1 <= num <= max_steps for num in selected):
            return sorted(set(selected))
    except ValueError:
        pass
    
    return None

def select_steps(steps):
    """Let user select which steps to run."""
    print("\n" + "="*70)
    print("Available Pipeline Steps:")
    print("="*70)
    
    for step_num, description, _, _ in steps:
        print(f"  {step_num:2d}. {description}")
    
    print("\n" + "="*70)
    print("Options: 'all' | '1,3,5' | '3-7' | 'skip1,skip3'")
    print("="*70)
    
    while True:
        choice = input("\nEnter your choice: ").strip()
        selected = parse_step_selection(choice, len(steps))
        
        if selected:
            return selected
        
        print("Invalid input. Please try again.")

def main():
    """Run the complete pipeline."""
    start_time = datetime.now()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                   SCF Color Naming Pipeline                           ║
║                   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}                         ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    logging.info("Starting SCF Color Naming Pipeline")
    
    # Create required directories
    create_directories()
    
    # Define pipeline steps
    steps = [
        (1, "Evaluate Multi-language SCF Models", "src.evaluation.opt_eval", []),
        (2, "Predict Multi-language Colour Names for J slices", "src.prediction.get_predictions_for_Jslices", []),
        (3, "Generate J slices", "src.visualization.Jslices_viz", []),
        (4, "Munsell Colour Analysis (British English)", "src.analysis.munsell", []),
        (5, "Munsell Visualization", "src.visualization.munsell_viz", []), 
        (6, "Run Baseline Models", "src.analysis.max_min_bound", []),           
        (7, "Probability Matrix Processing", "src.analysis.get_redcuedPCw", []),    
        (8, "Cross-language Translation Analysis", "src.analysis.get_translations", ["--all-pairs"]),
        (9, "Get Focal Colours", "src.analysis.process_translations", []),
        (10, "Perceptual Metric Evaluation", "src.analysis.perceptual_metric", []),
        (11, "Lexical Metric Evaluation", "src.analysis.lexical_metric", []),
    ]
    
    # Select steps to run
    selected_step_numbers = select_steps(steps)
    selected_steps = [step for step in steps if step[0] in selected_step_numbers]
    
    if not selected_steps:
        print("No steps selected. Exiting.")
        return
    
    print(f"\nRunning {len(selected_steps)} step(s): {[s[0] for s in selected_steps]}\n")
    
    # Execute selected steps
    successful = 0
    failed_steps = []
    durations = []
    
    for step_num, description, module_path, args in selected_steps:
        success, duration = run_step(step_num, description, module_path, args)
        durations.append((step_num, duration))
        
        if success:
            successful += 1
        else:
            failed_steps.append((step_num, description))
            response = input(f"\nStep {step_num} failed. Continue? (y/n): ")
            if response.lower() != 'y':
                logging.info("Pipeline stopped by user.")
                break
    
    # Summary
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                        PIPELINE COMPLETE                              ║
║                   Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}                         ║
║                   Duration: {str(total_time).split('.')[0]}                                  ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Successful: {successful:2d}/{len(selected_steps):2d}                                                  ║
║  Failed:     {len(failed_steps):2d}/{len(selected_steps):2d}                                                  ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    if failed_steps:
        logging.error("Failed steps:")
        for step_num, description in failed_steps:
            logging.error(f"  - Step {step_num}: {description}")
    
    logging.info(f"Pipeline finished. Total time: {total_time}")

if __name__ == "__main__":
    main()