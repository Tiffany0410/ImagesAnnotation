import os
import json
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import torch
import argparse

def process_image(image_filename, directory, global_args, annotations):
    image_id = os.path.splitext(image_filename)[0]
    if image_id not in annotations:  # Skip if already processed
        image_file = os.path.join(directory, image_filename)
        model_args = type('Args', (), {
            "model_path": global_args['model_path'],
            "model_base": None,
            "model_name": get_model_name_from_path(global_args['model_path']),
            "query": global_args['prompt'],
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
        output = eval_model(model_args)
        torch.cuda.empty_cache()  # Free unused memory
        return image_id, output
    return None

def generate_annotations(directory, output_file, batch_size=10):
    global_args = {
        'model_path': "liuhaotian/llava-v1.5-7b",
        'prompt': "What is in this image?"
    }

    annotations = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            annotations.update(json.load(f))

    image_filenames = [f for f in os.listdir(directory) if f.endswith(".jpg")]

    processed_count = 0
    for i in range(0, len(image_filenames), batch_size):
        batch_filenames = image_filenames[i:i + batch_size]
        for image_filename in batch_filenames:
            result = process_image(image_filename, directory, global_args, annotations)
            if result:
                image_id, output = result
                annotations[image_id] = output
                processed_count += 1
                if processed_count % 100 == 0:
                    with open(output_file, 'w') as f:
                        json.dump(annotations, f, indent=4)
                    print(f"Saved progress at {processed_count} images.")

    # Final save for all remaining data
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4)
    print("All images processed and data saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and generate annotations.")
    parser.add_argument("images_directory", type=str, help="The directory where images are stored")
    parser.add_argument("output_json_file", type=str, help="The file where annotations will be saved")
    args = parser.parse_args()

    generate_annotations(args.images_directory, args.output_json_file, batch_size=10)
