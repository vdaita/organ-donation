import gymnasium as gym
from typing import Optional, Dict, Any, Tuple
import numpy as np
from blood_type_encode import encode, blood_types, decode, blood_type_donate_to, blood_type_receive_from
from utils import generate_priority_scores
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx

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

        self.reset(reset_patients=True)
    
    def reset(self, seed: Optional[int] = None, reset_patients=False) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        if reset_patients:
            for pair in range(self.num_pairs):
                patient_type = np.random.choice(blood_types)         
                donor_type = np.random.choice(blood_types)
                encoded_patient = encode(patient_type, donor_type)
                self.patients[pair] = encoded_patient

        self.matched_patients = np.zeros(self.num_pairs, dtype=np.int8)
        self.current_selection = np.zeros(self.num_pairs, dtype=np.int8)
        self.steps_completed = 0

        observation = self._get_observation()
        info = {}
        return observation, info
        
    def _print_env(self):
        for i in range(self.num_pairs):
            patient, donor = decode(self.patients[i])
            print(f"Patient {i + 1}: {patient} (Donor: {donor})")

    def _get_observation(self):
        return {
            "patients": self.patients,
            "matched_patients": self.matched_patients,
            "current_selection": self.current_selection
        }
    
    def _validate(self, elements):
        valid_cycles, matched_pairs = self.optimized_top_trading_cycle(elements)
        return np.sum(matched_pairs) == len(elements)
    
    def optimized_top_trading_cycle(self, elements=None):
        if elements is None:
            elements = self.patients

        num_pairs = len(elements)
        remaining_pairs = set(range(num_pairs))
        valid_cycles = []
        matched_pairs = np.zeros(num_pairs, dtype=np.int8)

        while remaining_pairs:
            G = nx.DiGraph()
            G.add_nodes_from(remaining_pairs)
            
            # Add edges as before
            for i in remaining_pairs:
                patient_i_type, _ = decode(elements[i])
                for j in remaining_pairs:
                    if i != j:
                        _, donor_j_type = decode(elements[j])
                        if patient_i_type in blood_type_donate_to[donor_j_type]:
                            G.add_edge(i, j)
            
            # Look for a single cycle instead of all cycles
            try:
                # Find a single cycle efficiently
                cycle_found = False
                for node in remaining_pairs:
                    try:
                        # Find a cycle containing this node with limited length
                        cycle = nx.find_cycle(G, source=node, orientation='original')
                        cycle_nodes = [edge[0] for edge in cycle]
                        cycle_found = True
                        break
                    except nx.NetworkXNoCycle:
                        continue
                    
                if not cycle_found:
                    break
                    
                # Process cycle as before
                valid_cycles.append(cycle_nodes)
                for pair_id in cycle_nodes:
                    matched_pairs[pair_id] = 1
                    remaining_pairs.remove(pair_id)
                    
            except Exception as e:
                print(f"Error finding cycles: {e}")
                break
        
        return valid_cycles, matched_pairs

    def step(self, action, should_print=False):
        # print("Action: ", action)
        pair_idx = action
        reward = 0

        selected_size = np.sum(self.current_selection)

        if should_print:
            print("Action: ", action, " Current Selection: ", self.current_selection, " Matched: ", self.matched_patients)

        if pair_idx == self.num_pairs or selected_size == self.max_cycle + 1:
            # print("Loop completed: ", self.steps_completed)
            if not pair_idx == self.num_pairs:
                self.current_selection[pair_idx] = 1

            elements = self.patients[np.where(self.current_selection == 1)]

            if self._validate(elements):
                reward = len(elements)
                self.matched_patients[np.where(self.current_selection == 1)] = 1
            else:
                reward = -0.01
            
            self.current_selection = np.zeros_like(self.current_selection)
            if should_print:
                print("     Loop Completed - Reward: ", reward)
        else:
            if self.current_selection[pair_idx] == 1 or self.matched_patients[pair_idx] == 1:
                reward = -0.01
            else:
                self.current_selection[pair_idx] = 1

        self.steps_completed += 1

        return self._get_observation(), reward, (self.steps_completed == self.max_steps or np.sum(self.matched_patients) == self.num_pairs), False, {}
        
    def render(self):
        """Visualize the current state"""
        pass