import gymnasium as gym
from typing import Optional, Dict, Any, Tuple
import numpy as np
from blood_type_encode import encode, blood_types, decode, blood_type_donate_to
from utils import generate_priority_scores
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class PairedOrganDonationEnv(gym.Env):
    def __init__(self, num_pairs: int = 5, in_features: int = 25, max_steps: int = 32):
        self.num_pairs = num_pairs
        self.feature_dim = in_features
        self.max_steps = max_steps
        
        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "patients": gym.spaces.Box(low=0, high=1, shape=(num_pairs, self.feature_dim), dtype=np.int8),
            "matched_patients": gym.spaces.MultiBinary(num_pairs),
            "current_selection": gym.spaces.MultiBinary(num_pairs)
        })

        self.patients = np.zeros((num_pairs, self.feature_dim), dtype=np.int8)
        self.matched_patients = np.zeros(num_pairs, dtype=np.int8)
        self.current_selection = np.zeros(num_pairs, dtype=np.int8)

        self.action_space = gym.spaces.Discrete(num_pairs + 1)
        self.steps_completed = 0
    
        self.max_cycle = 8

        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        for pair in range(self.num_pairs):
            patient_type = np.random.choice(blood_types)
            donor_1_type = np.random.choice(blood_types)
            donor_2_type = np.random.choice(blood_types)
            patient_organ = np.random.choice(["kidney", "liver"])
            encoded_patient = encode(patient_type, patient_organ, donor_1_type, donor_2_type)
            self.patients[pair] = encoded_patient

        self.matched_patients = np.zeros(self.num_pairs, dtype=np.int8)
        self.current_selection = np.zeros(self.num_pairs, dtype=np.int8)
        self.steps_completed = 0

        observation = self._get_observation()
        info = {}
        return observation, info
        
    def _get_observation(self):
        return {
            "patients": self.patients,
            "matched_patients": self.matched_patients,
            "current_selection": self.current_selection
        }
    
    def _validate(self, elements):
        # Each patient has the option of their donor being kidney or dual_donor liver
        for mapping_int in range(pow(2, len(elements))):
            mapping = np.binary_repr(mapping_int, width=len(elements))
            mapping = [int(x) for x in mapping] 
            donor_assignments = ["kidney" if x == 1 else "liver" for x in mapping]
            
            # Create pools of available donors by blood type
            kidney_donors = []
            liver_donors = []
            
            # Track each patient's need
            patients = []
            
            # Collect all donors and patients from the current mapping
            for donor_type, patient_data in zip(donor_assignments, elements):
                patient_type, organ, donor_1_type, donor_2_type = decode(patient_data)
                
                # Add to donor pools based on assignment
                if donor_type == "kidney":
                    kidney_donors.append(donor_1_type)
                else:  # donor_type == "liver"
                    liver_donors.append(donor_1_type)
                    liver_donors.append(donor_2_type)
                
                # Record patient needs
                patients.append((patient_type, organ))
            
            # Try to match each patient with appropriate donors
            all_matched = True
            used_kidney_donors = []
            used_liver_donors = []

            scored_patients = generate_priority_scores(patients, strategy="rarity")
            patients = [patient for patient, _ in scored_patients]
            
            for patient_type, organ, patient_idx in patients:
                if organ == "kidney":
                    # Find ONE compatible kidney donor
                    donor_found = False
                    for i, donor in enumerate(kidney_donors):
                        if i not in used_kidney_donors and patient_type in blood_type_donate_to[donor]:
                            used_kidney_donors.append(i)
                            donor_found = True
                            break
                    
                    if not donor_found:
                        all_matched = False
                        break
                        
                else:  # organ == "liver"
                    # Find TWO compatible liver donors
                    donor_count = 0
                    for i, donor in enumerate(liver_donors):
                        if i not in used_liver_donors and patient_type in blood_type_donate_to[donor]:
                            used_liver_donors.append(i)
                            donor_count += 1
                            if donor_count == 2:
                                break
                    
                    if donor_count < 2:
                        all_matched = False
                        break
            
            # If all patients were matched, this is a valid mapping
            if all_matched:
                return True
        
        return False
    
    def brute_force_solve(self):
        def generate_combinations(n, max_size):
            all_combinations = []
            for size in range(1, max_size + 1):
                all_combinations.extend(combinations(range(n), size))
            return all_combinations

        valid_solutions = []
        # Rename this variable to avoid shadowing the imported function
        possible_combinations = generate_combinations(self.num_pairs, self.max_cycle)
        
        for combo in possible_combinations:
            test_selection = np.zeros(self.num_pairs, dtype=np.int8)
            test_selection[list(combo)] = 1
            elements = self.patients[np.where(test_selection == 1)]
            
            if self._validate(elements):
                decoded = [decode(patient) for patient in elements]
                valid_solutions.append(decoded)
                
        print("Valid solutions found:")
        for i, solution in enumerate(valid_solutions):
            print(f"\nSolution {i + 1}:")
            for patient in solution:
                print(f"Patient blood: {patient[0]}, Organ: {patient[1]}, Donor1: {patient[2]}, Donor2: {patient[3]}")
        
        return valid_solutions

    def step(self, action):
        # print("Action: ", action)
        pair_idx = action
        reward = 0

        selected_size = np.sum(self.current_selection)

        if pair_idx == self.num_pairs or selected_size == self.max_cycle:
            # print("Loop completed: ", self.steps_completed)
            elements = self.patients[np.where(self.current_selection == 1)]
            self.current_selection = np.zeros_like(self.current_selection)
            if self._validate(elements):
                reward = np.sum(self.matched_patients)
            elif self.steps_completed == self.max_cycle:
                reward = -0.1
        else:
            if self.current_selection[pair_idx] == 1 or self.matched_patients[pair_idx] == 1:
                reward = -0.1
            else:
                self.current_selection[pair_idx] = 1

        self.steps_completed += 1

        return self._get_observation(), reward, (self.steps_completed == self.max_steps), False, {}
        
    def render(self):
        """Visualize the current state"""
        pass