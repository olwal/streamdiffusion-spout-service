"""
Utility functions for the StreamDiffusion daemon
"""
import numpy as np
from PIL import Image
from typing import Dict, Optional

def numpy_to_pil(numpy_img: np.ndarray) -> Image.Image:
    """
    Convert a numpy array to a PIL Image.
    
    Args:
        numpy_img: NumPy array representing an image
        
    Returns:
        PIL Image object
    """
    if numpy_img.shape[2] == 3:  # RGB
        return Image.fromarray(numpy_img.astype(np.uint8))
    else:  # RGBA
        return Image.fromarray(numpy_img.astype(np.uint8), 'RGBA')
        
def parse_lora_string(lora_string: str) -> Dict[str, float]:
    """
    Parse a LoRA string into a dictionary.
    
    Args:
        lora_string: String in format "name1:scale1,name2:scale2"
        
    Returns:
        Dictionary mapping LoRA names to scales
    """
    if not lora_string:
        return None
        
    lora_dict = {}
    for pair in lora_string.split(','):
        if ':' in pair:
            name, scale = pair.split(':')
            try:
                lora_dict[name.strip()] = float(scale.strip())
            except ValueError:
                print(f"Warning: Invalid LoRA scale '{scale}' for '{name}', skipping")
    
    return lora_dict
