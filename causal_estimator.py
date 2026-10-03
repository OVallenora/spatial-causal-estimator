import numpy as np
from sklearn.cluster import KMeans
from scipy import stats

class SpatialCausalSimulator:
    """
    Simulates a geographic randomized control trial (RCT) using spatial clustering 
    to estimate the Average Treatment Effect (ATE) of a localized intervention.
    """
    def __init__(self, n_points: int = 1000, n_clusters: int = 5):
        self.n_points = n_points
        self.n_clusters = n_clusters
        self.true_causal_effect = 15.0  # The hidden actual impact of the treatment
        
    def generate_observational_data(self):
        """Generates geographic coordinates and a baseline outcome metric."""
        np.random.seed(42)
        # Simulate geographic coordinates (e.g., Luchon Valley bounding box)
        self.latitudes = np.random.uniform(42.7, 42.9, self.n_points)
        self.longitudes = np.random.uniform(0.5, 0.7, self.n_points)
        
        # Baseline outcome (e.g., baseline recovery speed) with natural variance
        self.baseline_outcome = np.random.normal(loc=50.0, scale=10.0, size=self.n_points)
        
    def apply_spatial_treatment(self):
        """Uses K-Means clustering to partition zones and assign treatment."""
        coords = np.column_stack((self.latitudes, self.longitudes))
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(coords)
        
        # Pragmatic treatment assignment: Treat clusters 0 and 1, control the rest
        self.is_treated = np.isin(self.clusters, [0, 1])
        
        # Calculate final outcome: baseline + causal effect (if treated) + random noise
        noise = np.random.normal(0, 2.0, self.n_points)
        treatment_impact = np.where(self.is_treated, self.true_causal_effect, 0)
        self.final_outcome = self.baseline_outcome + treatment_impact + noise

    def run_monte_carlo_evaluation(self, iterations: int = 2000):
        """
        Evaluates the statistical significance of the causal effect across 
        Monte Carlo simulation loops.
        """
        estimated_effects = []
        
        for _ in range(iterations):
            # Bootstrap sample for robust estimation
            indices = np.random.choice(self.n_points, size=self.n_points, replace=True)
            sample_treated = self.final_outcome[indices][self.is_treated[indices]]
            sample_control = self.final_outcome[indices][~self.is_treated[indices]]
            
            # Estimate Average Treatment Effect (ATE) for this loop
            ate_estimate = np.mean(sample_treated) - np.mean(sample_control)
            estimated_effects.append(ate_estimate)
            
        mean_ate = np.mean(estimated_effects)
        ci_lower, ci_upper = np.percentile(estimated_effects, [2.5, 97.5])
        
        # Run independent T-test on full dataset
        treated_group = self.final_outcome[self.is_treated]
        control_group = self.final_outcome[~self.is_treated]
        t_stat, p_value = stats.ttest_ind(treated_group, control_group)
        
        return mean_ate, ci_lower, ci_upper, p_value

if __name__ == "__main__":
    print("Initializing Spatial Causal Simulator...")
    sim = SpatialCausalSimulator(n_points=1000, n_clusters=5)
    
    print("Generating geographic data...")
    sim.generate_observational_data()
    
    print("Applying K-means spatial clustering for treatment assignment...")
    sim.apply_spatial_treatment()
    
    print(f"Running 2,000 Monte Carlo loops to estimate Causal Impact (True Effect: +{sim.true_causal_effect})...")
    ate, ci_low, ci_high, p_val = sim.run_monte_carlo_evaluation(iterations=2000)
    
    print("\n--- CAUSAL INFERENCE RESULTS ---")
    print(f"Estimated Average Treatment Effect (ATE): +{ate:.2f}")
    print(f"95% Confidence Interval: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"P-Value (Statistical Significance): {p_val:.4e}")
    if p_val < 0.05:
        print("Conclusion: Treatment effect is statistically significant.")
