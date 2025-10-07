import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import seaborn as sns
import pandas as pd
from torchvision import transforms
import argparse
import pdb

# Create a singleton LPIPS model instance
_lpips_model = None

def get_lpips_model():
    """Get or create the singleton LPIPS model instance"""
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex')
    return _lpips_model

def find_trained_models(where):
    """Find all trained model directories"""
    output_dir = where
    model_dirs = []
    
    for dir_name in os.listdir(output_dir):
        if dir_name.startswith('controlnet'):
            model_dirs.append(os.path.join(output_dir, dir_name))
    
    return model_dirs

def load_validation_data():
    """Load all validation images and their prompts"""
    validation_dir = './mydataset_local/validation_large'
    train_dir = './mydataset_local/train3/images'
    json_file_path = './mydataset_local/train3/prompt_truncated.json'
    
    # Get all validation images
    validation_images = [os.path.join(validation_dir, filename) 
                        for filename in os.listdir(validation_dir) 
                        if filename.endswith('_pose.png')]
    
    if not validation_images:
        raise ValueError("No validation images found")
    
    print(f"\nFound {len(validation_images)} validation images")
    
    # Load prompts for all validation images
    validation_data = []
    with open(json_file_path, 'r') as file:
        prompts = [json.loads(line) for line in file]
    print(f"\nTotal prompts loaded: {len(prompts)}")

    
    for val_img in validation_images:
        # Find corresponding prompt
        img_name = os.path.basename(val_img)
        # Remove _pose.png to get the base name
        base_name = img_name.replace('_pose.png', '')
        print(f"\nLooking for prompt matching: {base_name}")
        
        matching_prompt = next((p for p in prompts if p['source'] == img_name), None)
        
        if matching_prompt:
            # Find corresponding real image in train folder
            real_img_name = base_name + '.jpg'  # Real images are JPG
            real_img_path = os.path.join(train_dir, real_img_name)
            
            if os.path.exists(real_img_path):
                validation_data.append({
                    'validation_image': val_img,
                    'real_image': real_img_path,
                    'prompt': matching_prompt['prompt']
                })
                print(f"Found matching pair: {img_name} -> {real_img_name}")
            else:
                print(f"Warning: Could not find real image for {img_name}")
        else:
            print(f"Warning: Could not find prompt for {img_name}")
    
    if not validation_data:
        raise ValueError("No valid validation data found after matching with prompts and real images")
    
    return validation_data


def generate_image(model_path, prompt, control_image, guidance, device="cuda", size = 512):
    """Generate image using a trained model"""
    # Load the trained ControlNet
    controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=torch.float32)
    
    # Load the base pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        controlnet=controlnet,
        torch_dtype=torch.float32  # Changed from float16 to float32
    ).to(device)
    
    # Disable safety checker
    pipe.safety_checker = None
    
    # Load and preprocess the control image
    control_image = Image.open(control_image)
    control_image = control_image.resize((512, 512))
    
    # Generate the image
    image = pipe(
        prompt,
        control_image,
        num_inference_steps=100,
        guidance_scale=guidance
    ).images[0]
    
    return image

def calculate_metrics(generated_image, real_image):
    """Calculate SSIM and PSNR between generated and real images"""
    # Convert images to numpy arrays
    gen_array = np.array(generated_image)
    real_array = np.array(real_image)
    
    # Ensure images are in the same format and size
    if gen_array.shape != real_array.shape:
        real_array = np.array(Image.fromarray(real_array).resize(gen_array.shape[:2][::-1]))
    
    # Calculate SSIM and PSNR
    ssim_value = ssim(gen_array, real_array, channel_axis=2)
    psnr_value = psnr(real_array, gen_array)
    
    # Convert to torch tensors for LPIPS
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    gen_tensor = transform(generated_image).unsqueeze(0)
    real_tensor = transform(real_image).unsqueeze(0)
    
    # Calculate LPIPS using the singleton model
    lpips_value = get_lpips_model()(gen_tensor, real_tensor).item()

    return ssim_value, psnr_value, lpips_value

def create_metrics_summary_figure(all_metrics, guidance):
    """Create a figure with boxplots for each metric grouped by batch size and gradient accumulation"""
    # Extract all metrics into a format suitable for plotting
    plot_data = {
        'ssim': {'values': [], 'bs': [], 'ga': [], 'ylim': (0, 0.4)},
        'psnr': {'values': [], 'bs': [], 'ga': [], 'ylim': (8, 15)},
        'lpips': {'values': [], 'bs': [], 'ga': [], 'ylim': (0.4, 1)}
    }
    
    # Collect data from all images
    for image_data in all_metrics.values():
        for model_metrics in image_data['metrics']:
            # Extract batch size and gradient accumulation from model name
            model_params = model_metrics['model'].split(', ')
            bs = int(model_params[1].split(': ')[1])
            ga = int(model_params[2].split(': ')[1])
            
            # Add metrics to plot data
            plot_data['ssim']['values'].append(model_metrics['ssim'])
            plot_data['ssim']['bs'].append(bs)
            plot_data['ssim']['ga'].append(ga)
            
            plot_data['psnr']['values'].append(model_metrics['psnr'])
            plot_data['psnr']['bs'].append(bs)
            plot_data['psnr']['ga'].append(ga)
            
            plot_data['lpips']['values'].append(model_metrics['lpips'])
            plot_data['lpips']['bs'].append(bs)
            plot_data['lpips']['ga'].append(ga)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot each metric
    for idx, (metric_name, data) in enumerate(plot_data.items()):
        ax = axes[idx]
        
        # Create boxplot
        sns.boxplot(
            data=pd.DataFrame({
                'value': data['values'],
                'Batch Size': data['bs'],
                'Gradient Accumulation': data['ga']
            }),
            x='Batch Size',
            y='value',
            hue='Gradient Accumulation',
            ax=ax
        )
        
        # Set y-axis limits
        ax.set_ylim(data['ylim'])
        
        # Customize plot
        ax.set_title(f'{metric_name.upper()} Distribution')
        ax.set_ylabel(metric_name.upper())
        ax.set_xlabel('Batch Size')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'metrics_summary_guidance_{guidance}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_figure(validation_data, guidance, where = 'none'):
    """Create figures comparing results from all trained models for each validation image"""
    model_dirs = find_trained_models(where)
    
    
    if not model_dirs:
        raise ValueError("No trained models found")
    
    # Extract unique batch sizes and gradient accumulation steps
    batch_sizes = sorted(list(set([int(d.split('_bs')[1].split('_')[0]) for d in model_dirs])))
    grad_accums = sorted(list(set([int(d.split('_ga')[1].split('_')[0]) for d in model_dirs])))
    
    n_rows = len(grad_accums)  # One row per GA value
    n_cols = len(batch_sizes)  # One column per BS value
    
    # Store all metrics in a single dictionary
    all_metrics = {}
    
    # Process each validation image
    for idx, data in enumerate(validation_data):
        # Create figure with rows for GA values and columns for BS values
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
        
        # Handle different cases of axes array shape
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Load validation and real images
        control_image = Image.open(data['validation_image'])
        control_image = control_image.resize((512, 512))
        real_image = Image.open(data['real_image'])
        real_image = real_image.resize((512, 512))
        
        # Store metrics for this validation image
        image_metrics = []
        
        # Generate and plot images for each model
        for model_idx, model_dir in enumerate(model_dirs):
            try:
                # Extract model parameters from directory name
                params = model_dir.split('_')[-3:]
                lr = params[0].replace('lr', '')
                bs = params[1].replace('bs', '')
                ga = params[2].replace('ga', '')
                
                # Generate image
                generated_image = generate_image(model_dir, data['prompt'], data['validation_image'], guidance, size = size)
                
                # Calculate metrics
                ssim_value, psnr_value, lpips_value = calculate_metrics(generated_image, real_image)
                image_metrics.append({
                    'model': f"LR: {lr}, BS: {bs}, GA: {ga}",
                    'ssim': ssim_value,
                    'psnr': psnr_value,
                    'lpips': lpips_value
                })
                
                # When plotting, use the correct axes indexing
                row_idx = grad_accums.index(int(ga))
                col_idx = batch_sizes.index(int(bs))
                ax = axes[row_idx, col_idx]
                ax.imshow(np.array(generated_image))
                ax.set_title(f"BS: {bs}, GA: {ga}\nSSIM: {ssim_value:.3f}, PSNR: {psnr_value:.2f}")
                ax.axis('off')
                
            except Exception as e:
                print(f"Error processing model {model_dir}: {str(e)}")
                row_idx = grad_accums.index(int(ga))
                col_idx = batch_sizes.index(int(bs))
                axes[row_idx, col_idx].text(0.5, 0.5, f"Error: {str(e)}", 
                                   ha='center', va='center')
                axes[row_idx, col_idx].axis('off')
        
        # Hide empty subplots
        for idx in range(len(model_dirs), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'model_comparison_guidance_{guidance}_no_{idx+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store metrics for this image in the all_metrics dictionary
        image_name = os.path.basename(data['validation_image'])
        all_metrics[image_name] = {
            'prompt': data['prompt'],
            'metrics': image_metrics
        }
    
    # Save all metrics to a single file at the end
    with open(f'all_metrics_guidance_guidance_{guidance}.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    # Create summary figure with boxplots
    create_metrics_summary_figure(all_metrics, guidance)
    

def load_saved_metrics(guidance):
    """Load previously saved metrics from JSON file"""
    metrics_file = f'all_metrics_guidance_guidance_{guidance}.json'
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"No saved metrics found for guidance {guidance}")
    
    with open(metrics_file, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    where = 'D:/vit_512' #./outputs/16_8' #16_8'#  # './outputs/vit_thr_0.5_simple_caption'
    guidance = [4,5,6,7,8,9,10]
    validation_data = load_validation_data()    
    create_comparison_figure(validation_data, guidance, where=where)
    print(f"\nComparison figures and metrics saved for guidance {guidance}")

    