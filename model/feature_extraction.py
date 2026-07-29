"""
Feature Extraction Module for ClosetCraft
Uses ResNet50 pretrained model to extract image features for similarity matching
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from collections import defaultdict

class FashionFeatureExtractor:
    def __init__(self):
        """Initialize the ResNet50 model for feature extraction"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = self._get_transform()
        
    def _load_model(self):
        """Load pretrained ResNet50 and remove the classification layer"""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final fully connected layer to get feature vectors
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        return model
    
    def _get_transform(self):
        """Define image preprocessing transformations"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]     # ImageNet std
            )
        ])
    
    def extract_features(self, image_path):
        """
        Extract feature vector from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array of feature vector
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().flatten().cpu().numpy()
            
            return features
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def build_dataset_features(self, dataset_path='static/images', save_path='dataset_features.pkl'):
        """
        Extract and save features for all images in dataset
        
        Args:
            dataset_path: Path to folder containing dataset images
            save_path: Path to save extracted features
        """
        features_dict = {}
        image_files = [f for f in os.listdir(dataset_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(image_files)} images...")
        
        for filename in image_files:
            image_path = os.path.join(dataset_path, filename)
            features = self.extract_features(image_path)
            if features is not None:
                features_dict[filename] = features
                print(f"✓ Processed: {filename}")
        
        # Save features to pickle file
        with open(save_path, 'wb') as f:
            pickle.dump(features_dict, f)
        
        print(f"\n✓ Features saved to {save_path}")
        return features_dict
    
    def load_dataset_features(self, save_path='dataset_features.pkl'):
        """Load pre-computed dataset features"""
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def find_similar_items(self, query_image_path, dataset_features=None, top_k=5):
        """
        Find top K similar items from dataset
        
        Args:
            query_image_path: Path to query image
            dataset_features: Dictionary of dataset features
            top_k: Number of similar items to return
            
        Returns:
            List of (filename, similarity_score) tuples
        """
        # Extract query image features
        query_features = self.extract_features(query_image_path)
        if query_features is None:
            return []
        
        # Load dataset features if not provided
        if dataset_features is None:
            dataset_features = self.load_dataset_features()
            if dataset_features is None:
                return []
        
        # Calculate cosine similarity with all dataset images
        similarities = []
        for filename, features in dataset_features.items():
            similarity = cosine_similarity([query_features], [features])[0][0]
            similarities.append((filename, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def generate_outfit(self, uploaded_items):
        """
        Generate outfit combination from uploaded items
        
        Args:
            uploaded_items: Dictionary with keys 'top', 'bottom', 'shoes' etc.
                           Each value is the file path to the uploaded image
                           
        Returns:
            Dictionary with outfit recommendation
        """
        outfit_suggestion = {}
        
        # Extract features for each uploaded item
        item_features = {}
        for category, filepath in uploaded_items.items():
            features = self.extract_features(filepath)
            if features is not None:
                item_features[category] = features
        
        if len(item_features) < 2:
            return {"error": "Need at least 2 items to create an outfit"}
        
        # Simple matching logic: Suggest all uploaded items together
        # For a more sophisticated approach, you can add color matching here
        
        outfit_suggestion['recommended_outfit'] = list(item_features.keys())
        outfit_suggestion['items'] = uploaded_items
        
        # Add a simple compatibility score based on feature similarity
        total_similarity = 0
        comparisons = 0
        
        categories = list(item_features.keys())
        for i in range(len(categories)):
            for j in range(i+1, len(categories)):
                sim = cosine_similarity(
                    [item_features[categories[i]]], 
                    [item_features[categories[j]]]
                )[0][0]
                total_similarity += sim
                comparisons += 1
        
        if comparisons > 0:
            avg_similarity = total_similarity / comparisons
            if avg_similarity > 0.7:
                outfit_suggestion['compatibility'] = "Excellent match! These items go very well together."
            elif avg_similarity > 0.5:
                outfit_suggestion['compatibility'] = "Good match! This outfit looks cohesive."
            else:
                outfit_suggestion['compatibility'] = "Fair match. Consider adding accessories to complete the look."
        
        return outfit_suggestion

# Helper function for simple color extraction (for basic color matching)
def extract_dominant_color(image_path):
    """Extract dominant RGB color from an image (simplified version)"""
    from PIL import Image
    import numpy as np
    
    image = Image.open(image_path)
    image = image.resize((50, 50))
    pixels = np.array(image).reshape(-1, 3)
    
    # Simple average color (for basic matching)
    avg_color = pixels.mean(axis=0)
    return tuple(avg_color.astype(int))

def colors_match(color1, color2, threshold=50):
    """
    Check if two colors match based on RGB difference
    Simple complementary color logic
    """
    diff = np.abs(np.array(color1) - np.array(color2))
    return np.mean(diff) < threshold
if __name__ == "__main__":
    extractor = FashionFeatureExtractor()
    extractor.build_dataset_features()