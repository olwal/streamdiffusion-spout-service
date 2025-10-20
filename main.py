#!/usr/bin/env python3
"""
StreamDiffusion Spout Service
Main Entry Point
"""

import os
import sys
import time
import argparse
import threading
from pathlib import Path

# Add src/ to path for package imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Early argument parsing to get StreamDiffusion path before imports
def get_streamdiffusion_path():
    """Parse args early to get StreamDiffusion path"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--streamdiffusion-path', type=str, default=None)
    args, _ = parser.parse_known_args()
    return args.streamdiffusion_path

# Add StreamDiffusion utils to path BEFORE importing our modules
# Priority: 1. --streamdiffusion-path arg, 2. STREAMDIFFUSION_PATH env, 3. ../StreamDiffusion
def add_streamdiffusion_to_path(custom_path=None):
    """Add StreamDiffusion utils directory to Python path"""
    if custom_path:
        sd_path = Path(custom_path).resolve()
    elif 'STREAMDIFFUSION_PATH' in os.environ:
        sd_path = Path(os.environ['STREAMDIFFUSION_PATH']).resolve()
    else:
        # Default: look for sibling StreamDiffusion directory
        sd_path = (Path(__file__).parent.parent / 'StreamDiffusion').resolve()

    utils_path = sd_path / 'utils'
    if utils_path.exists():
        sys.path.insert(0, str(sd_path))
        return str(utils_path)
    else:
        print(f"Error: StreamDiffusion utils not found at: {utils_path}")
        print(f"Please set --streamdiffusion-path or STREAMDIFFUSION_PATH environment variable")
        sys.exit(1)

# Add StreamDiffusion to path early
try:
    custom_path = get_streamdiffusion_path()
    utils_path = add_streamdiffusion_to_path(custom_path)
except SystemExit:
    raise

# Now import our modules (after StreamDiffusion is in path)
from streamdiffusion_spout_service import config
from streamdiffusion_spout_service.osc_server import start_osc_server
from streamdiffusion_spout_service.diffusion_engine import start_diffusion_thread
from streamdiffusion_spout_service.utils import parse_lora_string

def main():
    """Main entry point for the service"""
    parser = argparse.ArgumentParser(description='StreamDiffusion Spout service')

    parser.add_argument('--streamdiffusion-path', type=str, default=None,
                        help='Path to StreamDiffusion repository (default: ../StreamDiffusion or $STREAMDIFFUSION_PATH)')
    parser.add_argument('--osc-ip', type=str, default=config.DEFAULT_OSC_IP,
                        help=f'OSC server IP (default: {config.DEFAULT_OSC_IP})')
    parser.add_argument('--osc-port', type=int, default=config.DEFAULT_OSC_PORT,
                        help=f'OSC server port (default: {config.DEFAULT_OSC_PORT})')
    parser.add_argument('--spout-in', type=str, default=config.DEFAULT_SPOUT_RECEIVER_NAME,
                        help=f'Spout receiver name (default: {config.DEFAULT_SPOUT_RECEIVER_NAME})')
    parser.add_argument('--spout-out', type=str, default=config.DEFAULT_SPOUT_SENDER_NAME,
                        help=f'Spout sender name (default: {config.DEFAULT_SPOUT_SENDER_NAME})')
    parser.add_argument('--model', type=str, default=config.DEFAULT_MODEL_ID,
                        help=f'Model ID or path (default: {config.DEFAULT_MODEL_ID})')
    parser.add_argument('--lora', type=str, default=None,
                        help='LoRA name:scale pairs (comma separated, e.g. "lora1:0.5,lora2:0.7")')
    parser.add_argument('--width', type=int, default=config.DEFAULT_WIDTH,
                        help=f'Image width (default: {config.DEFAULT_WIDTH})')
    parser.add_argument('--height', type=int, default=config.DEFAULT_HEIGHT,
                        help=f'Image height (default: {config.DEFAULT_HEIGHT})')
    parser.add_argument('--acceleration', type=str, default='xformers', 
                        choices=['none', 'xformers', 'tensorrt'],
                        help='Acceleration method (default: xformers)')
    parser.add_argument('--delta', type=float, default=0.5,
                        help='Delta multiplier of virtual residual noise (default: 0.5)')
    parser.add_argument('--verbose', type=int, default=config.DEFAULT_VERBOSE, choices=[0, 1, 2, 3],
                        help='Verbose level: 0=quiet, 1=startup/shutdown, 2=+OSC, 3=+prompts/frames (default: 1)')
    parser.add_argument('--quiet', action='store_true',
                        help='Set verbose to 0 (overrides --verbose)')
    
    args = parser.parse_args()

    # StreamDiffusion path already added at startup
    if config.verbose >= 2:
        print(f"Using StreamDiffusion utils from: {utils_path}")

    # Set verbose level based on arguments
    if args.quiet:
        config.verbose = 0
    else:
        config.verbose = args.verbose
    
    # Parse LoRA dict if provided
    lora_dict = parse_lora_string(args.lora) if args.lora else None
    
    # Create and start threads
    osc_thread = threading.Thread(
        target=start_osc_server,
        args=(args.osc_ip, args.osc_port)
    )
    
    sd_thread = threading.Thread(
        target=start_diffusion_thread,
        args=(
            args.model,
            lora_dict,
            args.width,
            args.height,
            args.spout_in,
            args.spout_out,
            args.acceleration,
            args.delta,
        )
    )
    
    try:
        if config.verbose >= 1:
            print("Starting StreamDiffusion Spout Service...")
            print("Note: You may see warnings from PyTorch/Diffusers during startup.")
            print("      Common warnings (safe to ignore):")
            print("      - 'No module named triton' - Triton is optional")
            print("      - 'resume_download is deprecated' - HuggingFace warning")
            print("      - 'safety checker disabled' - Expected for this setup")
            print("      - Config attribute warnings - Model compatibility notices")
            print()
        osc_thread.start()
        sd_thread.start()

        # Keep main thread alive for Ctrl+C handling
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        if config.verbose >= 1:
            print("\nShutting down...")
        config.exit_flag.set()

        # Wait for threads to finish
        osc_thread.join()
        sd_thread.join()

        if config.verbose >= 1:
            print("Shutdown complete")
        exit()

if __name__ == "__main__":
    main()
