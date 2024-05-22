import os
import json
import concurrent.futures
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import multiprocessing
import torch
import argparse

def process_images(batch_args):
    directory, batch_filenames, global_args, annotations = batch_args
    results = []
    
    for image_filename in batch_filenames:
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
            results.append((image_id, output))
    return results

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
    batches = [image_filenames[i:i + batch_size] for i in range(0, len(image_filenames), batch_size)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_images, (directory, batch, global_args, annotations)) for batch in batches]
        processed_count = 0
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            for image_id, output in results:
                annotations[image_id] = output
                processed_count += 1
                if processed_count % 200 == 0:
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

    multiprocessing.set_start_method('spawn')  # Set the start method for multiprocessing
    generate_annotations(args.images_directory, args.output_json_file, batch_size=10)

    # images_directory = "../../needle_felted/images"
    # output_json_file = "../../needle_felted/annotations.json"