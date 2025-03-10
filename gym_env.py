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

        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

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
        valid_cycles, matched_pairs = self.simple_top_trading_cycle(elements)
        return np.sum(matched_pairs) == len(elements)
    
    def simple_top_trading_cycle(self, elements=None):
        """
        Implementation of the Top Trading Cycle algorithm using NetworkX.
        
        Returns:
            List of cycles representing valid exchanges and the matched pairs
        """
        if elements is None:
            elements = self.patients

        num_pairs = len(elements)

        remaining_pairs = set(range(num_pairs))
        valid_cycles = []
        matched_pairs = np.zeros(num_pairs, dtype=np.int8)

        while remaining_pairs:
            # Build directed graph using NetworkX
            G = nx.DiGraph()
            G.add_nodes_from(remaining_pairs)
            
            # Add all possible compatibility edges
            for i in remaining_pairs:
                patient_i_type, _ = decode(elements[i])
                
                # Find all compatible donors (not just the first one)
                for j in remaining_pairs:
                    if i != j:
                        _, donor_j_type = decode(elements[j])
                        if patient_i_type in blood_type_donate_to[donor_j_type]:
                            G.add_edge(i, j)
            
            # Try to find cycles in the graph
            try:
                # Look for simple cycles in the directed graph
                cycles = list(nx.simple_cycles(G))
                
                if not cycles:
                    break
                    
                # Take the shortest cycle (generally more likely to be feasible)
                cycle_nodes = min(cycles, key=len)
                
                # Add cycle to results and remove from remaining pairs
                valid_cycles.append(cycle_nodes)
                for pair_id in cycle_nodes:
                    matched_pairs[pair_id] = 1
                    remaining_pairs.remove(pair_id)
                    
            except nx.NetworkXNoCycle:
                break  # No more cycles found
            except Exception as e:
                print(f"Error finding cycles: {e}")
                break
        
        return valid_cycles, matched_pairs

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