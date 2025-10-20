"""
Shared configuration and global variables
"""
import queue
import threading

# Default configuration values
DEFAULT_OSC_IP = "127.0.0.1"
DEFAULT_OSC_PORT = 7000
DEFAULT_SPOUT_RECEIVER_NAME = "SourceImage"
DEFAULT_SPOUT_SENDER_NAME = "StreamDiffusion"
DEFAULT_MODEL_ID = "stabilityai/sd-turbo"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_VERBOSE = 1

# Global variables for sharing data between threads
prompt_queue = queue.Queue()
trigger_event = threading.Event()
exit_flag = threading.Event()
start_event = threading.Event()
stop_event = threading.Event()
spout_send_event = threading.Event()
spout_restart_event = threading.Event()

# Default prompt settings
current_prompt = "abstract shape"
current_negative_prompt = "low quality, bad quality, blurry, low resolution"

# Runtime variables for storing latest images
latest_input_image = None
latest_output_image = None

# Verbose output control
verbose = DEFAULT_VERBOSE
