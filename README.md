# Flickr8k CLIP Retrieval System

This project demonstrates how to build a multimodal retrieval system using the Flickr8k dataset and the CLIP model. The system computes image embeddings from raw Flickr8k images using CLIP and then performs text-based image retrieval. To save computation time, the image embeddings are computed once, saved to disk, and then loaded on subsequent runs.

# Run the project
python main.py