# explanation.py

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import re
import warnings
from sklearn.linear_model import Lasso

warnings.filterwarnings("ignore")
import numpy as np

def _scalar_score(output):
    arr = np.array(output)
    # 0-dim
    if arr.ndim == 0:
        return float(arr)
    # 1-dim
    if arr.ndim == 1:
        idx = np.argmax(arr) if arr.shape[0] > 1 else 0
        return float(arr[idx])
    # 2-dim
    if arr.ndim == 2:
        last = arr[-1]
        idx  = np.argmax(last) if last.shape[0] > 1 else 0
        return float(last[idx])
    raise ValueError(f"Cannot reduce array of shape {arr.shape} to scalar")

class Explanation:
    def __init__(self, current_sample, histo_samples, model_path, original_score, feature_names, group_sizes, target):
        self.current_sample = current_sample
        self.histo_samples = histo_samples
        self.model_path = model_path

        self.original_score = np.squeeze(original_score)
        if isinstance(self.original_score, (np.ndarray, list)):
            if np.ndim(self.original_score) == 0:
                self.original_score = self.original_score.item()
            elif np.ndim(self.original_score) == 1:
                self.original_score = self.original_score[0].item()
            else:
                raise ValueError(f"Unexpected original_score shape: {self.original_score.shape}")

        self.feature_names = feature_names
        self.group_sizes = group_sizes
        self.target = target
        self.relevant_history = []
        self.delta = 0.001
        self.step = 10
        self.new_input = []
        self.weighted_samples = []
        self.coef = []

    def search_proper_input(self):
        step = 0
        model = keras.models.load_model(self.model_path)

        for i in range(len(self.histo_samples)):
            if step <= self.step:
                self.new_input = self.histo_samples[-(i+1):]
                raw           = model.predict(self.new_input, batch_size=1)
                current_score = _scalar_score(raw)
                current_delta = abs(current_score - self.original_score)

                if current_delta <= self.delta:
                    return self.new_input
                step += 1
                i *= 2
            else:
                return self.current_sample

        return None

    def capture_relevant_history(self):
        model = keras.models.load_model(self.model_path)
        current_prediction = model.predict(self.current_sample, batch_size=1)
        raw           = model.predict(self.new_input, batch_size=1)
        current_score = _scalar_score(raw)


        difference = abs(current_score - self.original_score)
        if difference > self.delta:
            self.new_input = self.histo_samples
            self.new_input = self.search_proper_input()
        else:
            self.new_input = self.current_sample

    def weighted_sampling(self, num_samples):
        sampled_list = []

        for idx, input_value in enumerate(self.new_input):
            distance = np.abs(idx + 1)
            new_weight = distance / (distance + 1)

            for _ in range(num_samples):
                random_sample = np.zeros_like(input_value)

                np.random.seed()
                num_selected = max(1, int(new_weight * input_value.shape[0] / 4))
                selected_indices = np.random.choice(input_value.shape[0], size=num_selected, replace=False)

                random_sample[selected_indices] = input_value[selected_indices]
                sampled_list.append(random_sample)

        self.weighted_samples = np.array(sampled_list)
        self.weighted_samples = self.weighted_samples.reshape(-1, self.new_input.shape[1])


    def sparse_group_lasso(self):
        group_index = []
        for grp_i, sz in enumerate(self.group_sizes):
            group_index += [grp_i+1] * sz


        model = keras.models.load_model(self.model_path)

        # 1) get *raw* model outputs for each weighted sample
        if self.target == 'lstm':
            raw_preds = model.predict(self.weighted_samples.reshape(-1,1,len(self.feature_names)))
        elif self.target == 'kitsune':
            raw_preds = model.predict(self.weighted_samples.reshape(-1,len(self.feature_names)))
        else:
            raw_preds = model.predict(self.weighted_samples.reshape(-1,1,len(self.feature_names)),
                                    batch_size=1)

        # 2) reduce each output to a single float
        y = np.array([_scalar_score(r) for r in raw_preds])

        # 3) flatten your feature matrix
        X = self.weighted_samples.reshape(len(y), -1)

        # 4) run your sparse‐group‐lasso (here via plain Lasso)
        lasso = Lasso(alpha=0.01)
        lasso.fit(X, y)

        # 5) store your coefficients
        self.coef = [lasso.coef_]


    def visualization(self, probs, group_sizes, group_names, feature_names, **kwargs):
        # Convert probs to 1D array
        probs = np.asarray(probs).flatten()
        num_features = len(feature_names)

        # Assign weights from probabilities (truncate or pad if necessary)
        if probs.size < num_features:
            # pad with zeros if fewer probs than features
            weights = np.zeros(num_features)
            weights[:probs.size] = probs
        else:
            # take first num_features probabilities
            weights = probs[:num_features]

        # Plotting
        fig, ax = plt.subplots()
        ax.bar(feature_names, weights)
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance (Class Probability)')
        ax.set_title('Feature Importance from Model Predictions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        weights = self.coef[0]
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

        plt.figure(figsize=(15, 5))  # Fixed size for better graph
        cmap = plt.colormaps.get_cmap("coolwarm")
        colors = cmap(weights)
        feature_index = 0

        for group_size, group_name in zip(group_sizes, group_names):
            group_weights = weights[feature_index:feature_index + group_size]
            group_colors = colors[feature_index:feature_index + group_size]
            group_labels = feature_names[feature_index:feature_index + group_size]

            for i, (weight, color, label) in enumerate(zip(group_weights, group_colors, group_labels)):
                plt.bar(feature_index + i, weight, color=color)
                plt.plot([feature_index + i, feature_index + group_size // 2], [weight, 1.2 * max(weights)], color='grey', linestyle='-')

            plt.text(feature_index + group_size // 2, 1.3, group_name, ha='center', va='top')
            feature_index += group_size
        return

        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right', fontsize=10)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.yticks([])

        plt.ylim(0, 1.5)
        plt.title("Feature Importance for Current Sample", fontsize=14)
        plt.tight_layout()
        plt.show()
