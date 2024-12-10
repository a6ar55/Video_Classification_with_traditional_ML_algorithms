import os
import pickle
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch.nn as nn

# File paths for saving features
TRAIN_FEATURES_FILE = "train_hmdb51_features.pkl"
VAL_FEATURES_FILE = "val_hmdb51_features.pkl"

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

# Feature Extraction Function for Video Clips
# Feature Extraction Function for Video Clips
def extract_features(dataset, feature_extractor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels = [], []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        for idx, (video, label) in enumerate(tqdm(dataloader, desc="Extracting Features")):
            video = video.squeeze(0).to(device)  # Shape: [T, H, W, C]
            frame_features = []

            # Process each frame individually
            for frame in video:
                frame = frame.permute(2, 0, 1).float() / 255.0  # Normalize to range [0, 1]
                frame = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame)  # Normalize
                frame = frame.unsqueeze(0)  # Add batch dimension [1, C, H, W]
                frame_features.append(feature_extractor(frame).view(-1))

            # Average the features across the sequence
            video_features = torch.mean(torch.stack(frame_features), dim=0)
            features.append(video_features.cpu().numpy())
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

# Train and Evaluate Models
def train_and_evaluate_models(train_features, train_labels, val_features, val_labels):
    results = {}

    # Define models to train
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=20, random_state=42, use_label_encoder=False, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=20, random_state=42),
        "SVM": SVC(kernel='linear', probability=True, random_state=42)
    }

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(train_features, train_labels)
        print(f"Evaluating {model_name}...")
        val_predictions = model.predict(val_features)
        report = classification_report(val_labels, val_predictions, output_dict=True)
        results[model_name] = report
        print(f"Classification Report for {model_name}:")
        print(classification_report(val_labels, val_predictions))

    return results

# Custom Dataset Class
# Custom Dataset Class
class CustomVideoDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None, frames_per_clip=16):
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.data = []

        # Parse the annotation file
        with open(annotation_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                file_path, label = row
                self.data.append((os.path.join(root_dir, file_path), label))

        # Create a mapping of class labels to integers
        self.classes = {label: idx for idx, label in enumerate(sorted(set(row[1] for row in self.data)))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        video, _, _ = read_video(video_path, pts_unit='sec')  # Read video frames

        # Select `frames_per_clip` frames evenly spaced across the video
        frame_indices = torch.linspace(0, video.shape[0] - 1, self.frames_per_clip).long()
        video_frames = video[frame_indices]

        if self.transform:
            # Apply transform only if it's not `ToTensor`
            video_frames = torch.stack([self.transform(frame) if not isinstance(frame, torch.Tensor) else frame for frame in video_frames])

        label_idx = self.classes[label]
        return video_frames, label_idx


# Dataset and Transform with Augmentations
print("Setting up dataset and transformations...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths
root_dir = 'out'  # Adjust this to your dataset directory
annotation_file = 'annotations.csv'

# Dataset
dataset = CustomVideoDataset(annotation_file=annotation_file, root_dir=root_dir, transform=transform, frames_per_clip=16)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

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
