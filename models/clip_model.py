import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import os


def load_clip_model(model_name, device):
    """
    Load the CLIP model and processor, then move the model to the given device.
    Args:
        model_name (str): Name of the pre-trained CLIP model.
        device (str): Device to load the model on ("cuda" or "cpu").
    Returns:
        tuple: (model, processor) for CLIP.
    """
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    return model, processor

def compute_image_embedding(image, model, processor, device):
    """
    Compute the CLIP embedding for a single image.
    Args:
        image (PIL.Image): Image loaded via PIL.
        model: The CLIP model.
        processor: The CLIP processor.
        device (str): Device ("cuda" or "cpu").
    Returns:
        torch.Tensor: Normalized image embedding.
    """
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    # Normalize the feature vector.
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu()

def compute_clip_embeddings(image_paths, model, processor, device):
    """
    Compute CLIP image embeddings for all images in the list.
    Args:
        image_paths (list): List of image file paths.
        model: The CLIP model.
        processor: The CLIP processor.
        device (str): Device for computation.
    Returns:
        dict: Mapping of image filename to its normalized embedding.
    """
    embeddings = {}
    for image_path in tqdm(image_paths, desc="Computing image embeddings"):
        try:
            # Open the image and convert it to RGB.
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        embedding = compute_image_embedding(image, model, processor, device)
        image_filename = image_path.split(os.sep)[-1]
        embeddings[image_filename] = embedding
    return embeddings
