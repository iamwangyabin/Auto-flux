import os
from PIL import Image
import yaml
from captionimage import process_images_in_batches, download_and_load_model
import argparse

parser = argparse.ArgumentParser(description='Process images and generate YAML configuration')
parser.add_argument('--concept', type=str, required=True, help='Concept name')
parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')
parser.add_argument('--trigger_word', type=str, default='anime style', help='Trigger word for captions')
args = parser.parse_args()


data_root = args.data_root
concept = args.concept
trigger_word = args.trigger_word


def convert_and_rename_images(dataset_dir):
    images = os.listdir(dataset_dir)
    for idx, image_name in enumerate(images):
        image_path = os.path.join(dataset_dir, image_name)
        
        try:
            # Open the image and convert it to PNG
            img = Image.open(image_path)
            img = img.convert("RGB")
            
            # Save as PNG with a new name
            new_image_name = f"{str(idx + 1)}.png"
            new_image_path = os.path.join(dataset_dir, new_image_name)
            
            img.save(new_image_path, "PNG")

            # Delete the old image if it's not already PNG
            if image_name != new_image_name:
                os.remove(image_path)
        
        except (IOError, OSError) as e:
            print("Error processing", image_path, ":", e)
            os.remove(image_path)  # Remove the corrupted file if error


def add_trigger_word_to_captions(dataset_dir, trigger_word):
    # Define the unwanted strings to be removed
    unwanted_phrases = [
        'The image is an illustration of ',
        'The image is a digital illustration of '
    ]
    
    for txt_filename in os.listdir(dataset_dir):
        if txt_filename.endswith('.txt'):
            txt_path = os.path.join(dataset_dir, txt_filename)

            # Read the caption from the file
            with open(txt_path, 'r') as f:
                caption = f.read().strip()
            
            # Remove unwanted phrases
            for phrase in unwanted_phrases:
                caption = caption.replace(phrase, "")
            
            # Add trigger word to the beginning of the modified caption
            new_caption = f"{trigger_word}, {caption}"
            
            # Write the updated caption back to the file
            with open(txt_path, 'w') as f:
                f.write(new_caption)
                
                

# Directory where images are located
dataset_dir = os.path.join(data_root, concept)

# Step 1: Convert and rename images to PNG
convert_and_rename_images(dataset_dir)

# Step 2: Caption all images using the model
model, processor = download_and_load_model('microsoft/Florence-2-large')
process_images_in_batches(dataset_dir, model, processor, batch_size=1)

# Step 3: Add trigger word to all text files
add_trigger_word_to_captions(dataset_dir, trigger_word)



def generate_yaml(name, folder_path):
    data = {
        'job': 'extension',
        'config': {
            'name': name,
            'process': [
                {
                    'type': 'sd_trainer',
                    'training_folder': "output",
                    'device': "cuda:0",
                    'network': {
                        'type': "lora",
                        'linear': 16,
                        'linear_alpha': 16
                    },
                    'save': {
                        'dtype': "float16",
                        'save_every': 1000,
                        'max_step_saves_to_keep': 10
                    },
                    'datasets': [
                        {
                            'folder_path': folder_path,
                            'caption_ext': "txt",
                            'caption_dropout_rate': 0.05,
                            'shuffle_tokens': False,
                            'cache_latents_to_disk': True,
                            'resolution': [512, 768, 1024]
                        }
                    ],
                    'train': {
                        'batch_size': 1,
                        'steps': 10000,
                        'gradient_accumulation_steps': 1,
                        'train_unet': True,
                        'train_text_encoder': False,
                        'gradient_checkpointing': True,
                        'noise_scheduler': "flowmatch",
                        'optimizer': "adamw8bit",
                        'lr': 1e-4,
                        'ema_config': {
                            'use_ema': True,
                            'ema_decay': 0.99
                        },
                        'dtype': "bf16"
                    },
                    'model': {
                        'name_or_path': "black-forest-labs/FLUX.1-dev",
                        'is_flux': True,
                        'quantize': True,
                    },
                    'sample': {
                        'sampler': "flowmatch",
                        'sample_every': 2000,
                        'width': 1024,
                        'height': 1024,
                        'prompts': [
                            "anime style, woman with red hair, playing chess at the park, bomb going off in the background",
                            "anime style, a woman holding a coffee cup, in a beanie, sitting at a cafe",
                            "anime style, a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini",
                            "anime style, woman playing the guitar, on stage, singing a song, laser lights, punk rocker",
                        ],
                        'neg': "",
                        'seed': 42,
                        'walk_seed': True,
                        'guidance_scale': 4,
                        'sample_steps': 20
                    }
                }
            ]
        },
        'meta': {
            'name': name,
            'version': '1.0'
        }
    }

    return yaml.dump(data, default_flow_style=False)


yaml_content = generate_yaml(concept, dataset_dir)


with open(f'{concept}.yaml', 'w') as file:
    file.write(yaml_content)


