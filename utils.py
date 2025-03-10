import numpy as np

def generate_priority_scores(patients, strategy="random"):
    """
    Claude: Generate priority scores for patients using different strategies.
    
    Args:
        patients: List of patient data tuples (patient_type, organ, idx)
        strategy: String indicating which priority strategy to use
    
    Returns:
        List of (patient, score) tuples, where higher scores indicate higher priority
    """
    scores = []
    
    if strategy == "random":
        # Random priority
        for patient in patients:
            scores.append((patient, np.random.random()))
    elif strategy == "rarity":
        # Prioritize rare blood types
        rarity_scores = {
            'O': 4,
            'B': 3,
            'A': 2,
            'AB': 1,
        }
        
        for idx, patient_type in enumerate(patients):
            scores.append(((patient_type, idx), rarity_scores[patient_type]))
    
    # Sort by priority score (highest first) and return
    return sorted(scores, key=lambda x: x[1], reverse=True)
