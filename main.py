import os
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from cuml.ensemble import RandomForestClassifier as cuRF  # GPU-accelerated Random Forest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# File paths for saving features
TRAIN_FEATURES_FILE = "train12_features.pkl"
VAL_FEATURES_FILE = "val12_features.pkl"

# Dataset Class with Weighted Sampling
class RGBIRDepthVideoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, sequence_length=16):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.data = []

        print("Loading dataset annotations...")
        with open(annotation_file, 'r') as f:
            for line in f:
                folder, frame_count, label = line.strip().split()
                self.data.append((folder, int(frame_count), int(label)))
        print(f"Loaded {len(self.data)} samples from {annotation_file}.")

        # Calculate class weights for balancing
        labels = [item[2] for item in self.data]
        class_counts = np.bincount(labels)
        self.class_weights = 1.0 / class_counts
        self.sample_weights = [self.class_weights[label] for label in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder, frame_count, label = self.data[idx]
        rgb_frames, ir_frames, depth_frames = [], [], []

        for i in range(1, min(frame_count, self.sequence_length) + 1):
            rgb_img_path = os.path.join(self.root_dir, 'rgb_data', folder, f"{i:06d}.jpg")
            ir_img_path = os.path.join(self.root_dir, 'ir_data', folder, f"{i:06d}.jpg")
            depth_img_path = os.path.join(self.root_dir, 'depth_data', folder, f"{i:06d}.png")

            rgb_frames.append(Image.open(rgb_img_path).convert('RGB'))
            ir_frames.append(Image.open(ir_img_path).convert('L'))  # Grayscale
            depth_frames.append(Image.open(depth_img_path).convert('L'))  # Grayscale

        # Padding for sequence length
        while len(rgb_frames) < self.sequence_length:
            rgb_frames.extend(rgb_frames[:self.sequence_length - len(rgb_frames)])
            ir_frames.extend(ir_frames[:self.sequence_length - len(ir_frames)])
            depth_frames.extend(depth_frames[:self.sequence_length - len(depth_frames)])

        if self.transform:
            rgb_frames = [self.transform(frame) for frame in rgb_frames]
            ir_frames = [self.transform(frame.convert('RGB')) for frame in ir_frames]
            depth_frames = [self.transform(frame.convert('RGB')) for frame in depth_frames]

        rgb_frames = torch.stack(rgb_frames)
        ir_frames = torch.stack(ir_frames)
        depth_frames = torch.stack(depth_frames)

        return (rgb_frames, ir_frames, depth_frames), label

# Feature Extractor Function
def get_resnet50_feature_extractor():
    print("Initializing ResNet50 feature extractor...")
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove last layer (fc)
    feature_extractor.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
    print("ResNet50 initialized and moved to device:", device)
    return feature_extractor

# Feature Extraction Function
def extract_features(dataset, feature_extractor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels = [], []
    sampler = WeightedRandomSampler(weights=dataset.sample_weights, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    total_samples = len(dataloader)
    with torch.no_grad():
        for idx, ((rgb_frames, ir_frames, depth_frames), label) in enumerate(tqdm(dataloader, desc="Extracting Features")):
            rgb_features_list, ir_features_list, depth_features_list = [], [], []

            for frame_idx in range(rgb_frames.size(1)):  # Process each frame individually
                rgb_frame = rgb_frames[:, frame_idx, :, :, :].to(device)  # Shape: [batch, channels, height, width]
                ir_frame = ir_frames[:, frame_idx, :, :, :].to(device)
                depth_frame = depth_frames[:, frame_idx, :, :, :].to(device)

                # Extract features for each frame
                rgb_features_list.append(feature_extractor(rgb_frame).view(-1))
                ir_features_list.append(feature_extractor(ir_frame).view(-1))
                depth_features_list.append(feature_extractor(depth_frame).view(-1))

            # Average the features across the sequence
            rgb_features = torch.mean(torch.stack(rgb_features_list), dim=0)
            ir_features = torch.mean(torch.stack(ir_features_list), dim=0)
            depth_features = torch.mean(torch.stack(depth_features_list), dim=0)

            combined_features = torch.cat([rgb_features, ir_features, depth_features]).cpu().numpy()
            features.append(combined_features)
            labels.append(label.item())

    return np.array(features), np.array(labels)

# Save and Load Features Functions
def save_features(filename, features, labels):
    print(f"Saving features to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)
    print("Features saved successfully.")

def load_features(filename):
    print(f"Loading features from {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print("Features loaded successfully.")
    return data['features'], data['labels']

# Train and evaluate multiple models
def train_and_evaluate_models(train_features, train_labels, val_features, val_labels):
    results = {}

    # Define models to train
    models = {
        "Random Forest (GPU)": cuRF(n_estimators=300, max_depth=20, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=20, random_state=42, use_label_encoder=False, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=20, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42)
    }

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        if model_name == "Random Forest (GPU)":
            model.fit(train_features, train_labels)
        else:
            for _ in tqdm(range(1), desc=f"Training {model_name}"):  # Progress bar for training
                model.fit(train_features, train_labels)

        print(f"Evaluating {model_name}...")
        val_predictions = model.predict(val_features)
        report = classification_report(val_labels, val_predictions, output_dict=True)
        results[model_name] = report
        print(f"Classification Report for {model_name}:")
        print(classification_report(val_labels, val_predictions))

    return results

# Dataset and Transform with Augmentations
print("Setting up dataset and transformations...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet normalization
])

train_dataset = RGBIRDepthVideoDataset(
    root_dir='./dataset0/train',
    annotation_file='./dataset0/train/train_videofolder.txt',
    transform=transform
)

val_dataset = RGBIRDepthVideoDataset(
    root_dir='./dataset0/validation',
    annotation_file='./dataset0/validation/val_videofolder.txt',
    transform=transform
)

# Feature Extraction or Loading
feature_extractor = get_resnet50_feature_extractor()
if os.path.exists(TRAIN_FEATURES_FILE) and os.path.exists(VAL_FEATURES_FILE):
    train_features, train_labels = load_features(TRAIN_FEATURES_FILE)
    val_features, val_labels = load_features(VAL_FEATURES_FILE)
else:
    print("Extracting features for training data...")
    train_features, train_labels = extract_features(train_dataset, feature_extractor)
    save_features(TRAIN_FEATURES_FILE, train_features, train_labels)

    print("Extracting features for validation data...")
    val_features, val_labels = extract_features(val_dataset, feature_extractor)
    save_features(VAL_FEATURES_FILE, val_features, val_labels)

# Train and evaluate the models
results = train_and_evaluate_models(train_features, train_labels, val_features, val_labels)

# Print results summary
for model_name, report in results.items():
    print(f"\nSummary for {model_name}:")
    print(f"Accuracy: {report['accuracy']}")
    print(f"Class-wise metrics:")
    for class_id, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"  Class {class_id}: Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1-score: {metrics['f1-score']}")
