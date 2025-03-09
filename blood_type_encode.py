import torch

blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
blood_type_donate_to = {
    'A+': ['A+', 'AB+'],
    'A-': ['A+', 'A-', 'AB+', 'AB-'],
    'B+': ['B+', 'AB+'],
    'B-': ['B+', 'B-', 'AB+', 'AB-'],
    'AB+': ['AB+'],
    'AB-': ['AB+', 'AB-'],
    'O+': ['A+', 'B+', 'AB+', 'O+'],
    'O-': ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
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

def encode(patient_type: str, patient_organ: str,  donor_1_type: str, donor_2_type: str):
    organ_encoded = torch.tensor([1 if patient_organ == "kidney" else 0], dtype=torch.float)
    return torch.cat([organ_encoded, encode_patient(patient_type), encode_donor(donor_1_type), encode_donor(donor_2_type)])

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
    organ = "kidney" if tensor[0].item() == 1 else "other"
    patient_tensor = tensor[1:9]
    donor_1_tensor = tensor[9:17]
    donor_2_tensor = tensor[17:25]
    
    patient_type = decode_patient_encoding(patient_tensor)
    donor_1_type = decode_donor_encoding(donor_1_tensor)
    donor_2_type = decode_donor_encoding(donor_2_tensor)
    return patient_type, organ, donor_1_type, donor_2_type
