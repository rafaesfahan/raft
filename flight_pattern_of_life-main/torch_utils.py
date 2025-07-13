def normalize_tensor(tensor):
    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)  # Adding a small epsilon to avoid division by zero
    return normalized_tensor, min_val, max_val

def unnormalize_tensor(tensor, min_val, max_val):
    unnormalized_tensor = tensor * (max_val - min_val + 1e-8) + min_val
    return unnormalized_tensor