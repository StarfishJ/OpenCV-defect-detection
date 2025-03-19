# Jar Lid Defect Detection System

A defect detection system for jar lids based on OpenCV and machine learning, capable of automatically identifying whether jar lids are intact or damaged.

![Alt text](./Screenshot.png) 

## Features

- Supports multiple image formats (JPG, JPEG, PNG)
- Uses various image feature extraction methods (HOG features, color histogram, edge features)
- Employs Random Forest Classifier for defect detection
- Provides intuitive visualization interface
- Supports batch image testing
- Displays prediction confidence levels

## System Requirements

- Python 3.6+
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Tkinter (Python standard library)

## Installation

1. Clone or download this project to your local machine
2. Install required Python packages:
```bash
pip install opencv-python numpy pandas scikit-learn matplotlib
```

## Data Preparation

1. Create a `train data` folder in the project root directory
2. Place training images in the `train data` folder
3. Prepare the annotation file `jarlids_annots.csv` with the following columns:
   - filename: Image filename
   - region_attributes: JSON format annotation information containing "type" field ("damaged" or "intact")

## Usage

1. Run the program:
```bash
python defect_detection.py
```

2. The program will automatically:
   - Load training data
   - Extract image features
   - Train the model
   - Display model evaluation results
   - Show prediction results for part of the test set

3. Testing new images:
   - Select images to test in the file selection dialog
   - Multiple images can be selected simultaneously (hold Ctrl key for multiple selection)
   - The program will display prediction results and confidence levels for each image

## Output Description

- The program will display:
  - Size of training, validation, and test sets
  - Model accuracy on each dataset
  - Prediction results and confidence levels for test images

## Notes

1. Ensure training images are of good quality, avoid blurry or severely distorted images
2. Image names in the annotation file must exactly match the actual filenames
3. It is recommended to use images of uniform size for optimal results

## Potential Improvements

1. Add more feature extraction methods
2. Optimize model parameters
3. Add batch processing functionality
4. Support more image formats
5. Add result export functionality
6. Optimize user interface

## Troubleshooting

If you encounter any issues, please check:
1. Python environment is properly configured
2. All required packages are installed
3. Data directory structure is correct
4. Annotation file format meets requirements

## License

MIT License 