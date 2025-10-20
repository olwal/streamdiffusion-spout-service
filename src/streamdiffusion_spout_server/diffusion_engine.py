"""
StreamDiffusion engine for image processing - Fixed for Spout Buffer Mode and PIL Image Handling
"""
import time
import numpy as np
from typing import Dict, Optional, Literal
import queue
import PIL
import warnings
import torch
import torchvision.transforms as T

# Filter out specific warnings
warnings.filterwarnings("ignore", message="Passing `image` as torch tensor with value range in")

from utils.wrapper import StreamDiffusionWrapper

from . import config
from .utils import numpy_to_pil
from .spout_handler import SpoutReceiver, SpoutSender

import SpoutGL

# Define image transformation for preprocessing
transform = T.Compose([
    T.ToTensor(),  # Convert PIL Image to tensor
])

def setup_stream_diffusion(
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]] = None,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    guidance_scale: float = 1.2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    seed: int = 2,
):
    """
    Initialize StreamDiffusion model.
    
    Args:
        model_id_or_path: Model ID or path
        lora_dict: Dictionary mapping LoRA names to scales
        width: Image width
        height: Image height
        acceleration: Acceleration method
        use_denoising_batch: Whether to use denoising batch
        guidance_scale: Guidance scale
        cfg_type: CFG type
        seed: Random seed
        
    Returns:
        Initialized StreamDiffusionWrapper
    """
    if guidance_scale <= 1.0:
        cfg_type = "none"

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[22, 32, 45],
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
    )

    return stream

# Initialize the prompt cache - store as global variable
prompt_cache = {}

def update_prompt_without_reset(stream, new_prompt, new_negative_prompt, delta, guidance_scale=None):
    """
    Updates the prompt without full model reset to avoid brown frames
    Includes efficient prompt caching to avoid redundant processing
    
    Args:
        stream: The StreamDiffusion instance
        new_prompt: The new prompt to use
        new_negative_prompt: The new negative prompt to use
        delta: The delta noise scale factor
        guidance_scale: Optional override for guidance scale (uses stream's value if None)
    """
    global prompt_cache
    
    # Create cache key from prompt and negative prompt
    cache_key = f"{new_prompt}||{new_negative_prompt}"
    
    # Check if prompt is already cached
    if cache_key in prompt_cache:
        encoder_output = prompt_cache[cache_key]
        if config.verbose >= 3:
            print(f"Using cached prompt: {new_prompt}")
    else:
        # Encode the prompt if not cached
        if config.verbose >= 3:
            print(f"Encoding new prompt: {new_prompt}")
        encoder_output = stream.pipe.encode_prompt(
            prompt=new_prompt,
            device=stream.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=new_negative_prompt,
        )
        # Cache the result for future use
        prompt_cache[cache_key] = encoder_output
        
        # Limit cache size to prevent memory issues (keep last 10 prompts)
        if len(prompt_cache) > 10:
            oldest_key = next(iter(prompt_cache))
            prompt_cache.pop(oldest_key)
    
    # Update only the prompt embeddings, not the entire model state
    stream.prompt_embeds = encoder_output[0].repeat(stream.batch_size, 1, 1)
    
    # Update guidance scale if provided
    if guidance_scale is not None:
        stream.guidance_scale = guidance_scale
    
    # If guidance scale is > 1.0, handle the classifier-free guidance
    if stream.guidance_scale > 1.0 and (
        stream.cfg_type == "initialize" or stream.cfg_type == "full"
    ):
        if encoder_output[1] is not None:  # Check if uncond_embeddings exists
            uncond_embeddings = encoder_output[1].repeat(stream.batch_size, 1, 1)
            stream.prompt_embeds = torch.cat([uncond_embeddings, stream.prompt_embeds], dim=0)
    
    # Scale the noise with delta but preserve the noise pattern
    stream.stock_noise *= delta
    
    # No timestep updates or other pipeline resets

def start_diffusion_thread(
    model_id,
    lora_dict,
    width,
    height,
    spout_receiver_name,
    spout_sender_name,
    acceleration,
    delta,
):
    """
    Thread function for diffusion processing.
    
    Args:
        model_id: Model ID or path
        lora_dict: Dictionary mapping LoRA names to scales
        width: Image width
        height: Image height
        spout_receiver_name: Spout receiver name
        spout_sender_name: Spout sender name
        acceleration: Acceleration method
        delta: Delta multiplier of virtual residual noise
    """
    # Initialize StreamDiffusion
    if config.verbose >= 1:
        print(f"Initializing StreamDiffusion with model: {model_id}")
    stream = setup_stream_diffusion(
        model_id_or_path=model_id,
        lora_dict=lora_dict,
        width=width,
        height=height,
        acceleration=acceleration,
    )

    # Initialize with default prompt
    if config.verbose >= 1:
        print(f"Preparing StreamDiffusion with prompt: '{config.current_prompt}'")
    stream.prepare(
        prompt=config.current_prompt,
        negative_prompt=config.current_negative_prompt,
        num_inference_steps=50,
        guidance_scale=1.2,
        delta=delta,
    )
    
    time.sleep(.5)
    

    # Initialize Spout
    receiver = SpoutReceiver(spout_receiver_name, width, height)
    sender = SpoutSender(spout_sender_name, width, height)

    # Create a black image for warmup
    if config.verbose >= 1:
        print("Warming up StreamDiffusion...")
    black_image = np.zeros((height, width, 3), dtype=np.uint8)
    black_pil = PIL.Image.fromarray(black_image)

    # Warmup with a proper black image
    for _ in range(stream.batch_size - 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stream.preprocess_image(black_pil)  # Just to initialize

    for _ in range(5):
        stream.preprocess_image(black_pil)  # Just to initialize

    if config.verbose >= 1:
        print("--------------------")
        print(f"Listening on Spout '{spout_receiver_name}' → sending to '{spout_sender_name}'")
        print("--------------------")
    
    # Counter for tracking frames
    frame_count = 0
    last_status_time = time.time()

    wait_frames = 0
    config.spout_send_event.set()

    while not config.exit_flag.is_set():
        # Check if Spout restart is requested
        if config.spout_restart_event.is_set():
            if config.verbose >= 1:
                print("Restarting Spout connections...")

            try:
                receiver.restart()
                sender.restart()
                config.spout_restart_event.clear()

                if config.verbose >= 1:
                    print("Spout connections restarted")

            except Exception as e:
                if config.verbose >= 1:
                    print(f"Spout restart failed: {e}")
                config.spout_restart_event.clear()
        
        # Check if we need to update the prompt
        try:
            new_prompt, new_negative_prompt = config.prompt_queue.get_nowait()

            if config.verbose >= 2:
                print(f"Updating prompt: {new_prompt[:40]}...")

            # Use the optimized update method with cache
            update_prompt_without_reset(stream.stream, new_prompt, new_negative_prompt, delta)

            config.prompt_queue.task_done()

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error updating prompt: {e}")
            if not config.prompt_queue.empty():
                config.prompt_queue.task_done()
        
        if config.stop_event.is_set():
            config.stop_event.clear()
            config.start_event.clear()

        # Check if we should process an image
        if config.trigger_event.is_set() or config.start_event.is_set():
            # Try to receive a frame
            input_image = receiver.receive_frame()

            if input_image is not None:
                frame_count += 1

                # Print status about received frame
                if config.verbose >= 3:
                    print(f"Processing frame #{frame_count}")

                if config.spout_send_event.is_set():
                    output_image = stream(stream.preprocess_image(input_image))
                    sender.send_frame(output_image)

            else:
                if config.verbose >= 3:
                    print("Trigger received but no input image available")
            
            # Reset trigger
            config.trigger_event.clear()

    # Cleanup
    receiver.close()
    sender.close()

    if config.verbose >= 1:
        print("Cleaned up diffusion thread resources... [OK]")
