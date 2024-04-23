import matplotlib.pyplot as plt
import numpy as np
import os, json

class DataVisualizer:
    def __init__(self, data):
        self.ids = list(data.keys())
        self.human_estimates = [data[key]["Human Estimate"] for key in self.ids]
        self.expected_values = [data[key]["Expected"] for key in self.ids]
        self.actual_values = [data[key]["Actual"] for key in self.ids]
        self.graph_dir = 'test_graphs'
        os.makedirs(self.graph_dir, exist_ok=True)

    def calculate_mape(self, true_values, predicted_values):
        true_values, predicted_values = np.array(true_values), np.array(predicted_values)
        return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

    def calculate_mae(self, true_values, predicted_values):
        true_values, predicted_values = np.array(true_values), np.array(predicted_values)
        return np.mean(np.abs(true_values - predicted_values))

    def plot_human_vs_expected(self):
        mape_score = self.calculate_mape(self.human_estimates, self.expected_values)
        mae_score = self.calculate_mae(self.human_estimates, self.expected_values)
        plt.figure(figsize=(10, 6))
        plt.scatter(self.human_estimates, self.expected_values, color='blue')
        plt.plot([min(self.human_estimates), max(self.human_estimates)], [min(self.human_estimates), max(self.human_estimates)], 'k--')
        plt.title(f"Human Estimates vs. Expected Values\nMAPE: {mape_score:.2f}%, MAE: {mae_score:.2f}")
        plt.xlabel("Human Estimates")
        plt.ylabel("Expected Values")
        plt.grid(True)
        plt.savefig(f"{self.graph_dir}/Human_vs_Expected.png")
        plt.close()

    def plot_human_vs_actual(self):
        mape_score = self.calculate_mape(self.human_estimates, self.actual_values)
        mae_score = self.calculate_mae(self.human_estimates, self.actual_values)
        plt.figure(figsize=(10, 6))
        plt.scatter(self.human_estimates, self.actual_values, color='green')
        plt.plot([min(self.human_estimates), max(self.human_estimates)], [min(self.human_estimates), max(self.human_estimates)], 'k--')
        plt.title(f"Human Estimates vs. Actual Values\nMAPE: {mape_score:.2f}%, MAE: {mae_score:.2f}")
        plt.xlabel("Human Estimates")
        plt.ylabel("Actual Values")
        plt.grid(True)
        plt.savefig(f"{self.graph_dir}/Human_vs_Actual.png")
        plt.close()

    def plot_actual_vs_expected(self):
        mape_score = self.calculate_mape(self.actual_values, self.expected_values)
        mae_score = self.calculate_mae(self.actual_values, self.expected_values)
        plt.figure(figsize=(10, 6))
        plt.scatter(self.actual_values, self.expected_values, color='red')
        plt.plot([min(self.actual_values), max(self.actual_values)], [min(self.actual_values), max(self.actual_values)], 'k--')
        plt.title(f"Actual Values vs. Expected Values\nMAPE: {mape_score:.2f}%, MAE: {mae_score:.2f}")
        plt.xlabel("Actual Values")
        plt.ylabel("Expected Values")
        plt.grid(True)
        plt.savefig(f"{self.graph_dir}/Actual_vs_Expected.png")
        plt.close()
        

json_path = 'test_results/estimate_comparison.json'
with open(json_path, 'r') as file:
    data = json.load(file)
comparison = DataVisualizer(data)

comparison.plot_human_vs_expected()
comparison.plot_human_vs_actual()
comparison.plot_actual_vs_expected()