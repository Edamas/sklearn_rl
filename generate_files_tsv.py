import os
import pandas as pd

project_root = "D:\\PROGRAMACAO\\sklearn_rl"
files_tsv_path = os.path.join(project_root, "files.tsv")

file_data = {}
seen_names = set()

for root, dirs, files in os.walk(project_root):
    # Exclude .git and __pycache__ directories
    dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__')]

    for file in files:
        absolute_path = os.path.join(root, file)
        relative_path = os.path.relpath(absolute_path, project_root)
        
        # Exclude the files.tsv itself
        if relative_path == "files.tsv":
            continue

        # Generate a simple, unique name
        base_name = os.path.splitext(file)[0]
        unique_name = base_name
        counter = 1
        while unique_name in seen_names:
            unique_name = f"{base_name}_{counter}"
            counter += 1
        
        seen_names.add(unique_name)
        file_data[unique_name] = relative_path

# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(file_data.items()), columns=['file_name', 'file_path'])

# Save to files.tsv
df.to_csv(files_tsv_path, sep='\t', index=False)

print(f"Updated {files_tsv_path} with {len(file_data)} files.")