import os
import pandas

import data
import metrics

levels = ['easy', 'medium', 'hard']
change = 'overlap_level'
base_path = f"datasets//{change}"
os.makedirs(base_path, exist_ok=True)

for overlap_level in levels:
    for another in levels:   
        config = data.build_synthetic_config(n_samples=another,
            n_features=another,
            n_classes=another,
            imbalance_level=another,
            overlap_level=overlap_level,
            noise_level=another,
            cluster_level=another,
            random_state=42)
    
    x, y = data.generate_synthetic_dataset(**config)
    
    df = x.copy()
    df["target"] = y

    filename = f"{base_path}/{change}_{overlap_level}_another_is_{another}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

