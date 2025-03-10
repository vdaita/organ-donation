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
            
    elif strategy == "urgency":
        # Prioritize liver patients (they need 2 donors)
        for patient_type, organ, idx in patients:
            urgency = 2.0 if organ == "liver" else 1.0
            scores.append(((patient_type, organ, idx), urgency))
            
    elif strategy == "rarity":
        # Prioritize rare blood types
        rarity_scores = {
            'O-': 5.0,  # Rarest and most restricted
            'AB-': 4.5,
            'B-': 4.0,
            'A-': 3.5,
            'O+': 3.0,
            'B+': 2.5,
            'A+': 2.0,
            'AB+': 1.0,  # Most common and least restricted
        }
        
        for idx, (patient_type, organ) in enumerate(patients):
            organ_factor = 1.5 if organ == "liver" else 1.0
            scores.append(((patient_type, organ, idx), rarity_scores[patient_type] * organ_factor))
    
    # Sort by priority score (highest first) and return
    return sorted(scores, key=lambda x: x[1], reverse=True)
