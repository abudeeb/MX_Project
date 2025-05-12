import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import re
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

class Explanation:
    def __init__(self, current_sample, histo_samples, model_path, original_score, feature_names, group_sizes, target):
        self.current_sample = current_sample
        self.histo_samples = histo_samples
        self.model_path = model_path

        # Ensure original_score is a scalar
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
                self.new_input = self.histo_samples[-(i + 1):]

                # Adjust input to fit Conv1D layer requirements: (batch_size, sequence_length, num_features)
                self.new_input = np.expand_dims(self.new_input, axis=-1)  # This adds the feature axis

                # Predict and ensure batch size of 1
                new_current_score = model.predict(self.new_input, batch_size=1)

                # Squeeze and ensure correct shape
                current_score = np.squeeze(new_current_score)  # Expecting shape (7,) for 7 classes

                if current_score.ndim == 1:
                    # If it's a 1D array (probabilities for all classes), pick the max value
                    current_score = np.max(current_score)  # Use max probability as the score
                elif current_score.ndim == 0:
                    # Scalar case, use directly
                    pass
                else:
                    raise ValueError(f"Unexpected prediction output shape: {current_score.shape}")

                # Calculate the difference between current score and the original score
                current_delta = abs(current_score - self.original_score)

                if current_delta <= self.delta:
                    return self.new_input  # Found proper input
                step += 1
                i *= 2  # Double the value of i
            else:
                return self.current_sample  # No proper input found within max steps

        return None  # Return None if no proper input is found within the step limit

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

        # Debugging: Check the weighted samples
        print("Weighted samples shape:", self.weighted_samples.shape)
        print("First 5 weighted samples:", self.weighted_samples[:5])

        # Normalize the weighted samples before applying any model
        scaler = StandardScaler()
        self.weighted_samples = scaler.fit_transform(self.weighted_samples)

        print("Scaled weighted samples:", self.weighted_samples[:5])  # Check scaled values

    def sparse_group_lasso(self):
        """
        Temporarily using raw model output probability-based importance for visualization
        """
        # Instead of RandomForest, directly using the output model probabilities or raw features
        # Compute raw feature importance by looking at how the individual features contribute to the prediction
        model = keras.models.load_model(self.model_path)
        raw_probs = model.predict(self.current_sample)

        # Debugging: Check the raw model output probabilities
        print("Raw model output probabilities:", raw_probs)

        # Calculate how much each feature (input) influences the model output (you can use any decision rule)
        feature_importance = np.abs(raw_probs - self.original_score)

        self.coef = [feature_importance.flatten()]  # Store the importance for visualization

    def visualization(self, group_sizes, group_names, feature_names):
        """
        Visualize the explanation with a better bar chart.
        """
        if len(self.coef) == 0:
            print("Error: Coefficients are not computed!")
            return

        weights = self.coef[0]

        # Debugging: Check raw weights before normalization
        print("Raw weights:", weights)

        # Normalize weights to range 0-1
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

        # Debugging: Check normalized weights
        print("Normalized weights:", weights)

        # Handle case where weights are all zero and avoid NaNs in plot
        if np.all(np.isnan(weights)):
            print("Warning: Weights are NaN. Cannot visualize.")
            return

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

        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right', fontsize=10)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.yticks([])

        plt.ylim(0, 1.5)
        plt.title("Feature Importance for Current Sample", fontsize=14)
        plt.tight_layout()
        plt.show()
