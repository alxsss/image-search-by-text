import torch

def search_images_by_text(query, model, processor, device, image_embeddings):
    """
    Search for images matching a given text query using CLIP.

    Args:
        query (str): The text query.
        model: The CLIP model.
        processor: The CLIP processor.
        device (str): Device for computation.
        image_embeddings (dict): Mapping of image filenames to their embeddings.

    Returns:
        list of tuples: Sorted list of (image_filename, similarity_score) tuples.
                        Higher scores indicate better matches.
    """
    # Process the text query through the CLIP text encoder.
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    # Normalize the text feature.
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu()

    similarities = {}
    # Compute cosine similarity between text and each image embedding.
    for image_filename, image_feat in image_embeddings.items():
        similarity = (text_features @ image_feat.T).item()
        similarities[image_filename] = similarity

    # Return results sorted by similarity (highest first).
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
