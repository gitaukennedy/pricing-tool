import os

for root, dirs, files in os.walk('data/raw'):
    for file in files:
        if file.endswith('.csv'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                row_count = sum(1 for _ in f) - 1
            print(f'{file}: {row_count:,} rows (Path: {path})')