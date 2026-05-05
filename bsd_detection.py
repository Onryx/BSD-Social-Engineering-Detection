import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util

class BSDDetector:
    def __init__(self, weights=None):
        # Default weights for T, C, B, N, and Kappa
        self.weights = weights if weights else [0.25, 0.25, 0.2, 0.2, 0.1]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scaler = MinMaxScaler()

    def calculate_temporal_score(self, timestamps):
        """Calculates timing regularity (T) using variance of intervals."""
        if len(timestamps) < 2: return 0
        intervals = np.diff(sorted(timestamps))
        # Lower variance in intervals suggests high synchronization
        regularity = 1 / (1 + np.var(intervals))
        return regularity

    def calculate_content_similarity(self, messages):
        """Calculates Content score (C) using SBERT embeddings."""
        if len(messages) < 2: return 0
        embeddings = self.model.encode(messages)
        sim_matrix = util.cos_sim(embeddings, embeddings)
        # Average similarity excluding self-comparison
        mask = np.ones(sim_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        return float(sim_matrix[mask].mean())

    def calculate_bsd_score(self, T, C, B, N, channels_used):
        """
        Computes the final Synchronization Score (S).
        Includes the Channel Covariate (Kappa inverse).
        """
        # Kappa inverse: higher value if attack spans multiple channels
        kappa_inv = 1 / (1 + (1 / len(channels_used)))
        
        scores = np.array([T, C, B, N, kappa_inv])
        final_score = np.dot(scores, self.weights)
        return final_score

# --- Example Usage ---
if __name__ == "__main__":
    detector = BSDDetector()
    
    # Mock data for a suspected coordinated campaign
    sample_timestamps = [1714900000, 1714900060, 1714900120] # Regular 60s intervals
    sample_messages = [
        "Please verify your bank account immediately.",
        "Your account requires urgent verification.",
        "Security alert: verify your banking credentials now."
    ]
    
    # 1. Get Dimension Scores
    t_score = detector.calculate_temporal_score(sample_timestamps)
    c_score = detector.calculate_content_similarity(sample_messages)
    b_score = 0.85  # Mock behavioral role sequence match
    n_score = 0.70  # Mock network clustering density
    
    # 2. Calculate Final BSD Score
    # Campaign spans 2 channels: Email and SMS
    s_total = detector.calculate_bsd_score(t_score, c_score, b_score, n_score, ['email', 'sms'])
    
    print(f"--- BSD Framework Detection Results ---")
    print(f"Synchronization Score (S): {s_total:.4f}")
    print(f"Status: {'COORDINATED ATTACK' if s_total > 0.70 else 'BENIGN'}")
      
