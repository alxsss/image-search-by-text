import os
import torch
import pickle
from config import IMAGE_DIR, MODEL_NAME, EMBEDDING_FILE
from utils.loader import load_images
from models.clip_model import load_clip_model, compute_clip_embeddings
from utils.helpers import search_images_by_text
from PIL import Image
from matplotlib import pyplot as plt   

def main():
    # Determine the device (GPU if available, else CPU).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the CLIP model and processor.
    model, processor = load_clip_model(MODEL_NAME, device)

    # Load image paths from the image directory.
    image_paths = load_images(IMAGE_DIR)
    print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

    # Check if the embeddings file exists.
    if os.path.exists(EMBEDDING_FILE):
        print("Loading precomputed image embeddings from disk...")
        with open(EMBEDDING_FILE, "rb") as f:
            image_embeddings = pickle.load(f)
    else:
        print("Computing image embeddings...")
        image_embeddings = compute_clip_embeddings(image_paths, model, processor, device)
        print("Saving computed embeddings to disk...")
        with open(EMBEDDING_FILE, "wb") as f:
            pickle.dump(image_embeddings, f)

    # Define a text query.
    query = "A group of people enjoying a sunny day outdoors"

    results = search_images_by_text(query, model, processor, device, image_embeddings)

    # Print the top matching images along with their similarity scores.
    print(f"\nTop matching images for query: '{query}'")
    for image_filename, score in results[:5]:
        print(f"{image_filename}: {score:.4f}")
    
        #Display images using PIL           
        image_path = os.path.join(IMAGE_DIR, image_filename)
        image = Image.open(image_path)
        plt.imshow(image)
        plt.title(f"{image_filename} ({score:.4f})")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
