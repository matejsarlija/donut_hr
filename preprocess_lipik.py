import json
from pathlib import Path

our_path = Path('racuni')
metadata_path = our_path.joinpath('key')

# Specify the path to your external JSON file
external_json_file = 'tasks_batch3.json'

# Load the JSON data from the external file with 'utf-8' encoding
with open(external_json_file, 'r', encoding='utf-8') as external_file:
    task_list = json.load(external_file)

# Iterate through each task entry
for task_data in task_list:
    # Check if 'metadata' is present in the task entry
    if 'metadata' in task_data:
        metadata_dict = task_data['metadata']
        metadata_filename = metadata_dict.get('filename', None)

        if metadata_filename:
            # Rest of your code remains the same
            annotations = task_data['response']['annotations']
    
    print(f"Metadata filename: {metadata_filename}")

    output_data = {}
    for annotation in annotations:
        label = annotation['label']
        text = annotation['text']
        output_data[label] = text

    # Create a JSON file with the extracted data
    output_filename = f"{metadata_filename}.json"
    with open(metadata_path.joinpath(output_filename), 'w', encoding='utf-8') as outfile:
        print(output_data)
        json.dump(output_data, outfile, ensure_ascii=False)

    print(f"Created JSON file: {output_filename}")
else:
    print("No valid metadata found in the JSON data.")
