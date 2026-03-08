import os
import json
import numpy as np

from preprocessing.normalize import normalize_strokes
from preprocessing.resample import resample
from preprocessing.feature_extraction import extract_features


class StrokeDatasetLoader:

    def __init__(self, data_dir, resample_points=64):
        self.data_dir = data_dir
        self.resample_points = resample_points

    def load_dataset(self):

        X = []
        y = []

        # Walk through all subdirectories to find JSON files
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:

                # ignore non-json files
                if not file.endswith(".json"):
                    continue

                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # preprocessing pipeline
                    points = normalize_strokes(data)

                    points = resample(points, self.resample_points)

                    features = extract_features(points)

                    X.append(features)

                    y.append(data["label"])

                except Exception as e:
                    print(f"Skipping {file}: {e}")

        X = np.array(X)
        y = np.array(y)

        print(f"Loaded {len(X)} samples")

        return X, y