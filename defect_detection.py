import cv2
import numpy as np
import os
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Set Chinese font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']  # For displaying Chinese labels
plt.rcParams['axes.unicode_minus'] = False  # For displaying negative signs


def select_test_images():
    """Open file selection dialog for test images"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Open file selection dialog with multiple selection support
    file_paths = filedialog.askopenfilenames(
        title='Select Test Images',
        initialdir='train data',  # Default to train data directory
        filetypes=[
            ('Image Files', '*.jpg;*.jpeg;*.png;*.JPG;*.JPEG;*.PNG'),
            ('All Files', '*.*')
        ]
    )
    
    if not file_paths:
        print("No images selected")
        return []
    
    return list(file_paths)


def load_data(data_dir, annot_file):
    """Load image data and labels"""
    images = []
    labels = []
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return np.array([]), np.array([])
    
    # Read annotation file
    try:
        df = pd.read_csv(annot_file)
    except Exception as e:
        print(f"Error: Cannot read annotation file - {str(e)}")
        return np.array([]), np.array([])
    
    # Process annotations grouped by filename
    for filename, group in df.groupby('filename'):
        img_path = os.path.join(data_dir, filename)
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Resize image to uniform size
        img = cv2.resize(img, (224, 224))
        images.append(img)
        
        # Determine if image contains damaged regions
        has_damaged = any(
            json.loads(attr)['type'] == 'damaged'
            for attr in group['region_attributes']
        )
        labels.append(1 if has_damaged else 0)
    
    if not images:
        print(f"Warning: No valid images found in {data_dir}")
        return np.array([]), np.array([])
    
    print(f"Loaded {len(images)} images from {data_dir}")
    return np.array(images), np.array(labels)


def extract_features(image):
    """Extract image features"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate HOG features
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray)
    
    # Calculate color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Calculate edge features
    edges = cv2.Canny(gray, 100, 200)
    edge_features = cv2.calcHist([edges], [0], None, [32], [0, 256])
    edge_features = cv2.normalize(edge_features, edge_features).flatten()
    
    # Combine features
    features = np.concatenate([hog_features.flatten(), hist, edge_features])
    return features


def test_images(model, image_paths):
    """Test multiple images"""
    if not image_paths:
        print("Error: No test images provided")
        return
    
    # Calculate subplot layout
    n_images = len(image_paths)
    n_cols = min(5, n_images)  # Maximum 5 images per row
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create figure
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"Error: Image file not found {image_path}")
            continue
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read image file {image_path}")
            continue
        
        # Resize image
        img_resized = cv2.resize(img, (224, 224))
        
        # Extract features and predict
        features = extract_features(img_resized)
        pred = model.predict([features])[0]
        pred_proba = model.predict_proba([features])[0]
        
        # Display results
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pred_text = 'Damaged' if pred == 1 else 'Intact'
        prob_text = f"Confidence: {pred_proba[1]:.2%}"
        plt.title(f"Prediction: {pred_text}\n{prob_text}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    # Load all data
    print("Loading data...")
    annot_path = os.path.join('train data', 'jarlids_annots.csv')
    all_images, all_labels = load_data('train data', annot_path)
    
    if len(all_images) == 0:
        print("Error: Data loading failed, "
              "please check data directory structure")
        return
    
    # Split into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, all_labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Extract features
    print("Extracting features...")
    train_features = np.array([extract_features(img) for img in X_train])
    val_features = np.array([extract_features(img) for img in X_val])
    test_features = np.array([extract_features(img) for img in X_test])
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(train_features, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    train_score = model.score(train_features, y_train)
    val_score = model.score(val_features, y_val)
    test_score = model.score(test_features, y_test)
    
    print(f"Training set accuracy: {train_score:.4f}")
    print(f"Validation set accuracy: {val_score:.4f}")
    print(f"Test set accuracy: {test_score:.4f}")
    
    # Visualize some prediction results
    plt.figure(figsize=(15, 5))
    for i in range(min(5, len(X_test))):
        plt.subplot(1, 5, i+1)
        img = X_test[i]
        pred = model.predict([extract_features(img)])[0]
        true_label = y_test[i]
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pred_text = 'Damaged' if pred == 1 else 'Intact'
        true_text = 'Damaged' if true_label == 1 else 'Intact'
        plt.title(f"Prediction: {pred_text}\nActual: {true_text}")
        plt.axis('off')
    plt.show()
    
    # Let user select test images
    print("\nPlease select images to test...")
    test_images_paths = select_test_images()
    
    if test_images_paths:
        print(f"Selected {len(test_images_paths)} images for testing")
        test_images(model, test_images_paths)
    else:
        print("No images selected, program ending")


if __name__ == "__main__":
    main() 