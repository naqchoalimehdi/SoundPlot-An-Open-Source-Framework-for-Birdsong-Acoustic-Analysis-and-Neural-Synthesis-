"""
Pattern detection and motif analysis.

Identifies recurring patterns, transitions, and motifs
in sequences of audio segments.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist


class PatternDetector:
    """
    Detect patterns and motifs in audio sequences.
    
    Analyzes sequences of cluster labels or features
    to find recurring patterns and transitions.
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize pattern detector.
        
        Args:
            similarity_threshold: Threshold for considering
                                 two segments as similar (0-1).
        """
        self.similarity_threshold = similarity_threshold
    
    def compute_transition_matrix(
        self,
        labels: np.ndarray,
        normalize: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute transition matrix from sequence of labels.
        
        Shows the probability of transitioning from one
        cluster/state to another.
        
        Args:
            labels: 1D array of cluster labels.
            normalize: If True, normalize rows to probabilities.
            
        Returns:
            Tuple of:
                - Transition matrix (n_states x n_states)
                - List of unique labels (state ordering)
        """
        # Remove noise labels (-1)
        valid_mask = labels >= 0
        labels = labels[valid_mask]
        
        if len(labels) < 2:
            return np.array([]), []
        
        unique_labels = sorted(set(labels))
        n_states = len(unique_labels)
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        
        # Count transitions
        matrix = np.zeros((n_states, n_states))
        
        for i in range(len(labels) - 1):
            from_state = label_to_idx[labels[i]]
            to_state = label_to_idx[labels[i + 1]]
            matrix[from_state, to_state] += 1
        
        # Normalize to probabilities
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            matrix = matrix / row_sums
        
        return matrix, unique_labels
    
    def find_ngrams(
        self,
        labels: np.ndarray,
        n: int = 2
    ) -> Dict[Tuple, int]:
        """
        Find n-gram patterns in label sequence.
        
        Args:
            labels: 1D array of cluster labels.
            n: Length of n-grams (2=bigrams, 3=trigrams).
            
        Returns:
            Dictionary mapping n-gram tuples to counts.
        """
        # Remove noise labels
        labels = labels[labels >= 0]
        
        if len(labels) < n:
            return {}
        
        ngrams = []
        for i in range(len(labels) - n + 1):
            ngram = tuple(labels[i:i + n])
            ngrams.append(ngram)
        
        return dict(Counter(ngrams))
    
    def find_repeated_sequences(
        self,
        labels: np.ndarray,
        min_length: int = 2,
        max_length: int = 5,
        min_occurrences: int = 2
    ) -> List[Dict]:
        """
        Find repeated sequences (motifs) in labels.
        
        Args:
            labels: 1D array of cluster labels.
            min_length: Minimum motif length.
            max_length: Maximum motif length.
            min_occurrences: Minimum times pattern must occur.
            
        Returns:
            List of dictionaries with motif info:
                - pattern: The sequence
                - count: Number of occurrences
                - positions: Starting indices
        """
        labels = labels[labels >= 0]
        
        motifs = []
        
        for length in range(min_length, max_length + 1):
            ngrams = self.find_ngrams(labels, length)
            
            for pattern, count in ngrams.items():
                if count >= min_occurrences:
                    # Find positions
                    positions = []
                    for i in range(len(labels) - length + 1):
                        if tuple(labels[i:i + length]) == pattern:
                            positions.append(i)
                    
                    motifs.append({
                        "pattern": list(pattern),
                        "length": length,
                        "count": count,
                        "positions": positions,
                    })
        
        # Sort by count (most frequent first)
        motifs.sort(key=lambda x: x["count"], reverse=True)
        
        return motifs
    
    def compute_self_similarity_matrix(
        self,
        features: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.
        
        Args:
            features: 2D feature array (n_samples x n_features).
            metric: Distance metric ("cosine", "euclidean").
            
        Returns:
            Similarity matrix (n_samples x n_samples).
        """
        # Compute distance matrix
        distances = cdist(features, features, metric=metric)
        
        # Convert to similarity
        if metric == "cosine":
            # Cosine distance is in [0, 2], convert to similarity
            similarity = 1 - (distances / 2)
        else:
            # For Euclidean, use exponential decay
            similarity = np.exp(-distances)
        
        return similarity
    
    def find_similar_segments(
        self,
        features: np.ndarray,
        query_idx: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find segments most similar to a query.
        
        Args:
            features: Feature array.
            query_idx: Index of query segment.
            top_k: Number of similar segments to return.
            
        Returns:
            List of (index, similarity_score) tuples.
        """
        similarity = self.compute_self_similarity_matrix(features)
        
        query_sim = similarity[query_idx]
        
        # Get top-k (excluding self)
        indices = np.argsort(query_sim)[::-1]  # Descending
        
        results = []
        for idx in indices:
            if idx != query_idx:
                results.append((int(idx), float(query_sim[idx])))
            if len(results) >= top_k:
                break
        
        return results
    
    def detect_motifs_by_similarity(
        self,
        features: np.ndarray,
        min_length: int = 3,
        max_length: int = 10
    ) -> List[Dict]:
        """
        Detect motifs based on feature similarity.
        
        Args:
            features: Feature array.
            min_length: Minimum motif length.
            max_length: Maximum motif length.
            
        Returns:
            List of detected motifs.
        """
        n_samples = len(features)
        similarity = self.compute_self_similarity_matrix(features)
        
        motifs = []
        
        # Simple approach: find diagonal patterns in similarity matrix
        for length in range(min_length, min(max_length + 1, n_samples // 2)):
            for i in range(n_samples - 2 * length + 1):
                for j in range(i + length, n_samples - length + 1):
                    # Check if segments are similar
                    diag_sim = [
                        similarity[i + k, j + k]
                        for k in range(length)
                    ]
                    avg_sim = np.mean(diag_sim)
                    
                    if avg_sim >= self.similarity_threshold:
                        motifs.append({
                            "position_a": i,
                            "position_b": j,
                            "length": length,
                            "similarity": float(avg_sim),
                        })
        
        # Sort by similarity
        motifs.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Remove overlapping motifs
        filtered = []
        used_positions = set()
        
        for motif in motifs:
            pos_a = set(range(motif["position_a"], 
                             motif["position_a"] + motif["length"]))
            pos_b = set(range(motif["position_b"],
                             motif["position_b"] + motif["length"]))
            
            if not (pos_a & used_positions or pos_b & used_positions):
                filtered.append(motif)
                used_positions.update(pos_a)
                used_positions.update(pos_b)
        
        return filtered
    
    def summarize_patterns(
        self,
        labels: np.ndarray
    ) -> Dict:
        """
        Generate a summary of patterns in the sequence.
        
        Args:
            labels: Cluster label sequence.
            
        Returns:
            Dictionary with pattern summary.
        """
        labels = labels[labels >= 0]
        
        if len(labels) == 0:
            return {
                "n_segments": 0,
                "n_unique_states": 0,
            }
        
        # Basic stats
        unique_states = set(labels)
        state_counts = Counter(labels)
        
        # Transition analysis
        transition_matrix, state_order = self.compute_transition_matrix(labels)
        
        # Find most common patterns
        bigrams = self.find_ngrams(labels, 2)
        trigrams = self.find_ngrams(labels, 3)
        
        most_common_bigram = max(bigrams.items(), key=lambda x: x[1]) if bigrams else None
        most_common_trigram = max(trigrams.items(), key=lambda x: x[1]) if trigrams else None
        
        return {
            "n_segments": int(len(labels)),
            "n_unique_states": int(len(unique_states)),
            "state_distribution": {int(k): int(v) for k, v in state_counts.items()},
            "most_common_bigram": {
                "pattern": [int(x) for x in most_common_bigram[0]],
                "count": int(most_common_bigram[1])
            } if most_common_bigram else None,
            "most_common_trigram": {
                "pattern": [int(x) for x in most_common_trigram[0]],
                "count": int(most_common_trigram[1])
            } if most_common_trigram else None,
            "transition_matrix": transition_matrix.tolist() if len(transition_matrix) > 0 else [],
            "state_order": [int(s) for s in state_order],
        }
