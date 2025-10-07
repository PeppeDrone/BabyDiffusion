import os
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# === CONFIGURATION ===
CHECKPOINTS_DIR = "D:/checks"  # Contains 'newborn/' and 'adults/'
VALIDATION_DIR = './mydataset_local/validation_examples'#large'
PROMPT_JSON_PATH = './mydataset_local/train3/prompt_truncated.json'
OUTPUT_ROOT = './visual_comparison'
OUTPUT_SUBDIR = 'three_panel_comparisons_250_2_8_fixed_baby_512'
NUM_SAMPLES = 200
RESOLUTION = 512
GUIDANCE_SCALE = 6
STEPS = 100

# === Prepare output directory ===
output_path = os.path.join(OUTPUT_ROOT, OUTPUT_SUBDIR)
os.makedirs(output_path, exist_ok=True)

# === Load prompt JSON ===
with open(PROMPT_JSON_PATH, 'r') as f:
    all_prompts = [json.loads(line) for line in f]
prompt_map = {entry['source']: entry['prompt'] for entry in all_prompts}

# === Gather validation images with prompts ===
validation_entries = []
for fname in sorted(os.listdir(VALIDATION_DIR)):
    if fname.endswith('_pose.png') and fname in prompt_map:
        validation_entries.append({
            'image_id': fname.replace('_pose.png', ''),
            'control_image_path': os.path.join(VALIDATION_DIR, fname),
            'prompt': prompt_map[fname]
        })
    if len(validation_entries) >= NUM_SAMPLES:
        break

# === Helper: Load model pipeline ===
def load_pipeline_0(model_dir):
    controlnet = ControlNetModel.from_pretrained(model_dir, torch_dtype=torch.float16, device='meta')
    controlnet = controlnet.to_empty(device=torch.device("cuda"))
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        controlnet=controlnet,
        torch_dtype=torch.float16, local_files_only=True
    ).to('cuda')
    pipe.safety_checker = None
    
    return pipe
def load_pipeline_1(model_dir):
    
    controlnet = ControlNetModel.from_pretrained(model_dir, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", controlnet=controlnet, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False
    ).to('cuda')
       
    return pipe

# === Load pipelines ===
print("🚀 Loading ControlNet models...")
pipe_newborn = load_pipeline_1(os.path.join(CHECKPOINTS_DIR, 'newborn'))
pipe_adults = load_pipeline_0(os.path.join(CHECKPOINTS_DIR, 'adults'))

# === Generation loop ===
print("🔄 Generating images and saving comparison panels...")
for idx, entry in enumerate(validation_entries):
    try:
        image_id = entry['image_id']
        # prompt = entry['prompt']
        prompt = "A baby on a bed"
        control_img = Image.open(entry['control_image_path']).resize((RESOLUTION, RESOLUTION))

        print(f"[{idx+1}/{NUM_SAMPLES}] Generating for: {image_id}")

        # Newborn generation
        img_newborn = pipe_newborn(
            prompt,
            image=control_img,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).images[0]

        # Adults generation
        img_adults = pipe_adults(
            prompt,
            image=control_img,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).images[0]

        # Plot and save comparison figure
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(control_img)
        axs[0].set_title('Pose')
        axs[1].imshow(img_newborn)
        axs[1].set_title('Newborn')
        axs[2].imshow(img_adults)
        axs[2].set_title('Adults')

        for ax in axs:
            ax.axis('off')

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, f"{image_id}_comparison.png"))
        plt.close(fig)

    except Exception as e:
        print(f"❌ Error processing {entry['image_id']}: {e}")

print(f"\n✅ All {NUM_SAMPLES} comparison figures saved in: {output_path}")
