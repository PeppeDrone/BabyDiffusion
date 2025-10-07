import json
import sys

def get_prompt_for_image(image_name, prompt_file="mydataset_local/train/prompt.json"):
    """
    Get the prompt associated with a specific image name from the prompt.json file.
    
    Args:
        image_name (str): The name of the image (e.g., 'segment_0893_029_frame_000150.jpg')
        prompt_file (str): Path to the prompt.json file
    
    Returns:
        str: The prompt associated with the image, or None if not found
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['target'] == image_name:
                    return data['prompt']
        return None
    except FileNotFoundError:
        print(f"Error: Could not find prompt file at {prompt_file}")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in prompt file")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_prompt.py <image_name> [prompt_file_path]")
        print("Example: python get_prompt.py segment_0893_029_frame_000150.jpg")
        sys.exit(1)
    
    image_name = sys.argv[1]
    prompt_file = sys.argv[2] if len(sys.argv) > 2 else "mydataset_local/train/prompt.json"
    
    prompt = get_prompt_for_image(image_name, prompt_file)
    
    if prompt:
        print(f"\nImage: {image_name}")
        print(f"Prompt: {prompt}")
    else:
        print(f"No prompt found for image: {image_name}")

if __name__ == "__main__":
    main() 