import subprocess
from itertools import product
import os
import json

def get_prompts_from_sources(json_file_path, source_images, whole):
    """
    Retrieves prompts from a JSON file based on provided source image filenames.

    Parameters:
    - json_file_path: Path to the JSON file.
    - source_images: List of source image filenames to find prompts for.

    Returns:
    - Dictionary mapping source image filenames to their prompts.
    """
    source_to_prompt = {}
    with open(json_file_path, 'r') as file:
        for line in file:
            obj = json.loads(line)
            validation_path = './mydataset_local\\validation_whole\\' if whole else './mydataset_local\\validation\\'
            if validation_path + obj['source'] in source_images:
                source_to_prompt[obj['source']] = obj['prompt']

    return source_to_prompt





# Fixed directories
root_dir = './mydataset_local'
whole = 0
if whole:
    train_dir = os.path.join(root_dir, 'train4') 
    validation_dir = os.path.join(root_dir, 'validation_whole')
else:
    train_dir = os.path.join(root_dir, 'train3') 
    validation_dir = os.path.join(root_dir, 'validation')
json_file_path = os.path.join(train_dir, 'prompt_truncated.json')
reference_bs = 8
reference_ga = 8
reference_steps = 1000 # in 512 tests this is 1000
total_samples = reference_bs * reference_ga * reference_steps



# Parameters to vary
learning_rates = [1e-5]
batch_sizes = [4,2]
grad_acc = [8,4]
res = 768
# Extract information for validation images
validation_images = [os.path.join(validation_dir, filename) 
                     for filename in os.listdir(validation_dir) 
                     if filename.endswith(('.jpg', '.png'))]
prompts = get_prompts_from_sources(json_file_path, validation_images, whole)
validation_prompts  = []
print('Validation images/prompts:')
for source_img in validation_images:
    prompt = prompts.get(source_img.split('\\')[-1], 'Prompt not found')
    print(f"{source_img}: {prompt}")
    validation_prompts.append(prompt)
    

### Execute training

# Iteration: for lr, bs, res in product(learning_rates, batch_sizes, resolutions):
for lr, bs, ga in product(learning_rates, batch_sizes, grad_acc):
    effective_bs = bs * ga
    adjusted_steps = int(total_samples // effective_bs)
    
    base_command = [
    "accelerate", "launch", "train_controlnet.py",
    "--pretrained_model_name_or_path=stabilityai/stable-diffusion-2-1",
    "--dataset_name=./mydataset_local/my_dataset_local.py",
    "--image_column=image",
    "--conditioning_image_column=conditioning_image",
    "--caption_column=text",
    "--mixed_precision=fp16",
    "--adam_beta1=0.9",
    "--adam_beta2=0.999",
    "--adam_epsilon=1e-8",
    "--adam_weight_decay=0.01",
    "--lr_scheduler=cosine",             # Add cosine scheduler
    "--lr_warmup_steps=500"              # Linear warm-up for first 500 steps
] + [f"--resolution={res}"] + [f"--max_train_steps={adjusted_steps}"]

    
    if whole:
        base_command.append("--train_data_dir=./mydataset_local/train4")
    else:
        base_command.append("--train_data_dir=./mydataset_local/train3")
    
    
    validation_image_args = sum([["--validation_image", img] for img in validation_images], [])
    validation_prompt_args = sum([["--validation_prompt", prompt] for prompt in validation_prompts], [])

    if whole:
        output_dir = f"./outputs/run/controlnet21_fullbody_BLIP2short_lr{lr}_bs{bs}_ga{ga}_res{res}"
    else:
        output_dir = f"./outputs/run/controlnet21_vit0.5_BLIP2short_lr{lr}_bs{bs}_ga{ga}_res{res}"


    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    command = base_command + validation_image_args + validation_prompt_args + [
        f"--output_dir={output_dir}",
        f"--learning_rate={lr}",
        f"--train_batch_size={bs}",
        f"--gradient_accumulation_steps={ga}",
        "--num_validation_images=1",
        "--validation_steps=100",  
        "--resume_from_checkpoint=./outputs/run/controlnet21_vit0.5_BLIP2short_lr1e-05_bs4_ga8_res1024/checkpoint-650",
        "--checkpointing_steps=100"
    ]    

    print(f"Learning rate: {lr}, Batch size: {bs}, Gradient Accumulation: {ga}")
    subprocess.run(command, check=True)
