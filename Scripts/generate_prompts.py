import argparse
import json
import os
from glob import glob
from pathlib import Path

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Update prompts using BLIP model')
    parser.add_argument('--input_dir', type=str, default='train3/images',
                      help='Directory containing input images')
    parser.add_argument('--prompt_file', type=str, default='train3/prompt.json',
                      help='Path to the prompt.json file')
    parser.add_argument('--model', type=str, 
                      choices=['blip_caption:base_coco',
                              'blip2_opt:caption_coco_opt2.7b',
                              'blip2_opt:caption_coco_opt6.7b',
                              'blip2_t5:pretrain_flant5xxl'],
                      default='blip2_t5:pretrain_flant5xxl',
                      help='BLIP model to use for captioning')
    return parser.parse_args()

def load_prompt_file(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_prompt_file(prompt_file, data):
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    args = parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory {args.input_dir} does not exist")
    
    # Check if prompt file exists
    if not os.path.exists(args.prompt_file):
        raise FileNotFoundError(f"Prompt file {args.prompt_file} does not exist")
    
    # Load prompt data
    prompt_data = load_prompt_file(args.prompt_file)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load BLIP model
    model_name, model_type = args.model.split(':')
    model, vis_processors, _ = load_model_and_preprocess(
        name=model_name,
        model_type=model_type,
        is_eval=True,
        device=device
    )
    
    # Get all image files
    image_files = glob(os.path.join(args.input_dir, '*.*'))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    
    # Process each image
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        
        # Find corresponding entry in prompt data
        entry = None
        for item in prompt_data:
            if item.get('image') == image_name:
                entry = item
                break
        
        if entry is None:
            print(f"Warning: No matching entry found for image {image_name}")
            continue
        
        # Load and process image
        try:
            raw_image = Image.open(image_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            
            # Generate caption
            prompt = "Describe the image in detail."
            caption = model.generate(
                samples={"image": image, "prompt": prompt},
                use_nucleus_sampling=True,
                temperature=0.7,
                top_p=0.9,
                max_length=50,
                repetition_penalty=1.2,
            )[0]
            
            # Update prompt in the entry
            entry['prompt'] = caption.strip()
            print(f"Updated prompt for {image_name}: {caption.strip()}")
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
    
    # Save updated prompt data
    save_prompt_file(args.prompt_file, prompt_data)
    print(f"Updated prompts saved to {args.prompt_file}")

if __name__ == "__main__":
    main() 