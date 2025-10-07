import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def extract_training_params(model_dir):
    """Extract batch size and gradient accumulation from model directory name"""
    dir_name = os.path.basename(model_dir)
    # Look for patterns like bs4_ga8 or similar in the directory name
    bs_match = re.search(r'bs(\d+)', dir_name)
    ga_match = re.search(r'ga(\d+)', dir_name)
    
    batch_size = int(bs_match.group(1)) if bs_match else 4  # default to 4 if not found
    grad_acc = int(ga_match.group(1)) if ga_match else 8    # default to 8 if not found
    
    return batch_size, grad_acc

def find_trained_models(output_dir='D:/vit'):#'./outputs/16_8'):#D:/vit_thr_0.5_simple_caption'):#):
    """Find all trained model directories"""
    logger.info(f"Searching for trained models in {output_dir}")
    model_dirs = []
    for dir_name in os.listdir(output_dir):
        if dir_name.startswith('controlnet'):
            model_dirs.append(os.path.join(output_dir, dir_name))
    logger.info(f"Found {len(model_dirs)} model directories")
    return model_dirs

def plot_training_losses():
    logger.info("Starting to plot training losses")
    # Get all model directories
    model_dirs = find_trained_models()
    n_models = len(model_dirs)
    
    # Force 2x2 layout
    n_cols = 2
    n_rows = 4
    
    logger.info(f"Creating figure with {n_rows} rows and {n_cols} columns")
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.flatten()
    
    # Reference values for step adjustment
    reference_bs = 8
    reference_ga = 8
    reference_steps = 1000
    total_samples = reference_bs * reference_ga * reference_steps
    
    # Plot each model's training loss and learning rate
    for idx, model_dir in enumerate(model_dirs):
        log_dir = os.path.join(model_dir, 'logs', 'train_controlnet')
        logger.info(f"Processing model {idx+1}/{n_models}: {os.path.basename(model_dir)}")
        
        try:
            # Get model's batch size and gradient accumulation
            batch_size, grad_acc = extract_training_params(model_dir)
            effective_bs = batch_size * grad_acc
            adjusted_steps = total_samples // effective_bs
            
            # Load the events file
            logger.debug(f"Loading events from {log_dir}")
            event_acc = event_accumulator.EventAccumulator(log_dir)
            event_acc.Reload()
            
            # Get loss and learning rate data
            losses = event_acc.Scalars('loss')
            lrs = event_acc.Scalars('lr')
            
            # Extract steps and values
            raw_steps = [entry.step for entry in losses]
            loss_values = [entry.value for entry in losses]
            lr_values = [entry.value for entry in lrs]
            
            # Calculate adjusted steps
            adjusted_step_values = [step * (adjusted_steps / reference_steps) for step in raw_steps]
            
            logger.debug(f"Loaded {len(raw_steps)} data points")
            
            # Find minimum loss point
            min_idx = np.argmin(loss_values)
            min_step = adjusted_step_values[min_idx]
            min_loss = loss_values[min_idx]
            
            # Create plot with two y-axes
            ax1 = axes[idx]
            ax2 = ax1.twinx()  # Create secondary y-axis
            
            # Plot loss on primary y-axis
            line1 = ax1.plot(adjusted_step_values, loss_values, 'b-', label='Loss')
            # Add X marker at minimum loss point
            ax1.scatter(min_step, min_loss, color='blue', marker='x', s=100, 
                       label=f'Min Loss: {min_loss:.4f}')
            ax1.set_xlabel('Adjusted Training Steps')
            ax1.set_ylabel('Loss', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Plot learning rate on secondary y-axis
            line2 = ax2.plot(adjusted_step_values, lr_values, 'r-', label='Learning Rate')
            ax2.set_ylabel('Learning Rate', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Force 0-1 scale on all axes
            ax1.set_ylim(0, 1)
            
            # Add title and grid
            model_name = os.path.basename(model_dir)
            ax1.set_title(f'Training Metrics: {model_name}\nBS={batch_size}, GA={grad_acc}, Eff_BS={effective_bs}')
            ax1.grid(True)
            
            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
            
        except Exception as e:
            logger.error(f"Error processing {model_dir}: {str(e)}")
            axes[idx].text(0.5, 0.5, f"Error loading data\n{str(e)}", 
                         ha='center', va='center', transform=axes[idx].transAxes)
    
    # Hide empty subplots if any
    for idx in range(len(model_dirs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = 'training_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Figure saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_training_losses()