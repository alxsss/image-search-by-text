import os

def load_images(image_dir):
    """
    Load image file paths from the given directory.
    Args:
        image_dir (str): Path to the folder containing images.
    Returns:
        list: List of full paths to image files.
    """
    # List files with typical image extensions.
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith(valid_extensions)
    ]
    return image_paths
