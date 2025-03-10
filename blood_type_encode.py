import torch

blood_types = ['A', 'B', 'AB', 'O']
blood_type_donate_to = {
    'A': ['A', 'AB'],
    'B': ['B', 'AB'],
    'AB': ['AB'],
    'O': ['A', 'B', 'AB', 'O']
}
blood_type_receive_from = {}
for donor, recipients in blood_type_donate_to.items():
    for recipient in recipients:
        if recipient not in blood_type_receive_from:
            blood_type_receive_from[recipient] = []
        blood_type_receive_from[recipient].append(donor)

def encode_donor(blood_type: str):
    encoding = [1 if bt in blood_type_donate_to[blood_type] else 0 for bt in blood_types]
    return torch.tensor(encoding, dtype=torch.float)

def encode_patient(blood_type: str):
    encoding = [1 if bt in blood_type_receive_from[blood_type] else 0 for bt in blood_types]
    return torch.tensor(encoding, dtype=torch.float)

def encode(patient_type: str, donor_type: str):
    return torch.cat([encode_patient(patient_type), encode_donor(donor_type)])

def decode_donor_encoding(tensor: torch.Tensor) -> str:
    for bt in blood_types:
        if torch.all(encode_donor(bt) == tensor):
            return bt
    return None

def decode_patient_encoding(tensor: torch.Tensor) -> str:
    for bt in blood_types:
        if torch.all(encode_patient(bt) == tensor):
            return bt
    return None

def decode(tensor: torch.Tensor):
    patient_tensor = tensor[:4]
    donor_tensor = tensor[4:8]
    
    patient_type = decode_patient_encoding(patient_tensor)
    donor_type = decode_donor_encoding(donor_tensor)
    return patient_type, donor_type
