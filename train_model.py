"""
Run this ONCE to train the KNN model and save it as model.pkl

Usage:
    python train_model.py --dataset /path/to/your/dataset

Dataset folder structure expected:
    dataset/
        Happy/
            img1.jpg
            img2.jpg
            ...
        Sad/
            img1.jpg
            ...
"""

import argparse
from ml_model import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--output', type=str, default='model.pkl', help='Output model file path')
    args = parser.parse_args()

    print(f"Training on dataset: {args.dataset}")
    accuracy, cm, _, _ = train_model(args.dataset, args.output)

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Confusion Matrix:")
    print(f"  True Negative (Sad→Sad):   {cm[0,0]}")
    print(f"  False Positive (Sad→Happy): {cm[0,1]}")
    print(f"  False Negative (Happy→Sad): {cm[1,0]}")
    print(f"  True Positive (Happy→Happy):{cm[1,1]}")
    print(f"{'='*50}")
    print(f"\nModel saved to: {args.output}")
    print(f"Now run: python app.py")
