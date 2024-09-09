def load_checkpoints(checkpoint_dir):
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth'):
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            checkpoint = torch.load(checkpoint_path)
            adjusted_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            checkpoints.append(adjusted_checkpoint)
    return checkpoints

def average_checkpoints(checkpoints):
    avg_state_dict = OrderedDict()
    num_checkpoints = len(checkpoints)

    for state_dict in checkpoints:
        for key, param in state_dict.items():
            if key not in avg_state_dict:
                avg_state_dict[key] = param.clone() / num_checkpoints
            else:
                avg_state_dict[key] += param / num_checkpoints

    return avg_state_dict

def load_model_with_avg_weights(model_class, avg_state_dict):
    model = model_class()  # Initialize your model class here
    model.load_state_dict(avg_state_dict)
    model.eval()
    return model

def make_predictions(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions
