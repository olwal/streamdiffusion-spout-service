# StreamDiffusion Spout Service

Real-time AI image generation service using StreamDiffusion, controllable via OSC and sharing textures via Spout.

## Overview

A real-time image generation service using StreamDiffusion, controllable via OSC (Open Sound Control) and communicating via Spout for GPU-accelerated texture sharing.

**Platform:** Windows only (requires Spout)

## Features

- **OSC Control**: Receive prompts and control commands via OSC
- **Spout I/O**: GPU-accelerated texture sharing for real-time communication
- **Prompt Caching**: Fast prompt switching without model reset
- **Real-time Generation**: Continuous img2img generation loop
- **Flexible Resolution**: Supports any resolution (default 512×512, configurable via command line)

## Architecture

```
Client Application (Processing, TouchDesigner, Cinder, etc.)
  ↓ Spout: "SourceImage" (input texture)
  ↓ OSC: /prompt "your prompt here"
StreamDiffusion Spout Service
  ↓ StreamDiffusion Pipeline (SD-Turbo)
  ↓ Spout: "StreamDiffusion" (generated image)
Client Application receives & displays
```

## Installation

### Prerequisites

- **Windows** (Spout is Windows-only)
- **Python 3.9+**
- **CUDA-capable GPU** (NVIDIA)

### Setup

**Recommended directory structure:** Clone both repositories as siblings:

```
your-workspace/
  ├── StreamDiffusion/                # Official StreamDiffusion repo
  └── streamdiffusion_spout_service/  # This service
```

**Steps:**

1. **Clone and install StreamDiffusion**

   ```bash
   # Clone StreamDiffusion library
   git clone https://github.com/cumulo-autumn/StreamDiffusion.git
   cd StreamDiffusion
   pip install .
   cd ..
   ```

2. **Clone this repository**

   ```bash
   git clone https://github.com/olwal/streamdiffusion-spout-service.git
   cd streamdiffusion_spout_service
   ```

3. **Install PyTorch with CUDA support**

   ```bash
   # For CUDA 11.8 (adjust cu118 to your CUDA version: cu117, cu121, etc.)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install remaining dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

After installation, run the service:

```bash
# If you cloned as siblings (recommended):
python main.py

# If StreamDiffusion is elsewhere, specify the path:
python main.py --streamdiffusion-path /path/to/StreamDiffusion

# On Windows, use forward slashes or quote backslashes:
python main.py --streamdiffusion-path ../StreamDiffusion
python main.py --streamdiffusion-path "..\StreamDiffusion"

# Or set environment variable:
export STREAMDIFFUSION_PATH=/path/to/StreamDiffusion
python main.py
```

**Send OSC commands** (from another terminal or application):
```
/prompt "a beautiful landscape"
/start
```

**Note:** If running directly from source without installing the package, Python automatically adds `src/` to the path.

## Usage

### Basic Usage

```bash
# Start with default settings
python main.py

# Or use the included batch file (Windows)
start.bat

# Custom settings
python main.py --model stabilityai/sd-turbo --width 512 --height 512 --osc-port 7000
```

### Command Line Options

```
--streamdiffusion-path PATH  Path to StreamDiffusion repo (default: ../StreamDiffusion or $STREAMDIFFUSION_PATH)
--osc-ip IP                  OSC server IP (default: 127.0.0.1)
--osc-port PORT              OSC server port (default: 7000)
--spout-in NAME              Spout receiver name (default: SourceImage)
--spout-out NAME             Spout sender name (default: StreamDiffusion)
--model MODEL                Model ID or path (default: stabilityai/sd-turbo)
--lora LORA                  LoRA name:scale pairs (e.g. "lora1:0.5,lora2:0.7")
--width W                    Image width (default: 512, can be any size)
--height H                   Image height (default: 512, can be any size)
--acceleration TYPE          Acceleration: none, xformers, tensorrt (default: xformers)
--delta FLOAT                Delta noise multiplier (default: 0.5)
--verbose LEVEL              Verbosity: 0=quiet, 1=startup, 2=+OSC, 3=+frames (default: 1)
--quiet                      Quiet mode (verbose=0)
```

## OSC Commands

Send OSC messages to control the service (default port: 7000):

### Prompt Control
```
/prompt "your prompt here"
```

### Generation Control
```
/start    or /s     - Start continuous generation
/stop     or /S     - Stop continuous generation
/trigger  or /t     - Trigger single generation
```

### Spout Control
```
/p      - Enable Spout output
/P      - Disable Spout output
/x      - Restart Spout connections
```

### Verbosity Control
```
/verbose 2    - Set verbosity level (0-3)
/v            - Cycle through verbosity levels
/von          - Verbose on (level 2)
/voff         - Quiet mode (level 0)
```

### Example: Sending OSC from Python

```python
from pythonosc import udp_client

client = udp_client.SimpleUDPClient("127.0.0.1", 7000)

# Set prompt and start generation
client.send_message("/prompt", "a beautiful mountain landscape, sunset")
client.send_message("/start", [])
```

### Example: Sending OSC from Command Line

Using `oscsend` (from liblo-tools):
```bash
oscsend localhost 7000 /prompt s "a beautiful landscape"
oscsend localhost 7000 /start
```

## Troubleshooting

### "No module named 'triton'" Warning

You may see this warning at startup:
```
A matching Triton is not available, some optimizations will not be enabled.
Error caught was: No module named 'triton'
```

**This is safe to ignore.** Triton is an optional NVIDIA optimization library. The daemon will work fine without it using xformers acceleration instead.

To suppress the warning (optional):
```bash
pip install triton
```

Note: Triton only works on Linux. On Windows, this warning is expected and harmless.

### Spout Connection Issues
```bash
# Restart Spout connections via OSC
# Send: /x

# Or check Spout sender list in your client application
```

### Prompt Not Updating
```bash
# Enable verbose mode to see OSC messages
python main.py --verbose 2

# Check OSC port matches your client application (default: 7000)
```

### GPU Out of Memory
```bash
# Use smaller model or reduce batch size
# Already using minimal frame_buffer_size=1 in diffusion_engine.py
```

### StreamDiffusion Not Found

If you get "StreamDiffusion utils not found", you have several options:

```bash
# Option 1: Use command-line argument
python main.py --streamdiffusion-path /path/to/StreamDiffusion

# Option 2: Set environment variable (persists for session)
export STREAMDIFFUSION_PATH=/path/to/StreamDiffusion
python main.py

# Option 3: Ensure sibling directory structure (default)
your-workspace/
  ├── StreamDiffusion/
  └── streamdiffusion_spout_service/
```

## Project Structure

```
streamdiffusion_spout_service/
├── src/
│   └── streamdiffusion_spout_service/  # Main package
│       ├── __init__.py
│       ├── config.py                  # Configuration & global state
│       ├── osc_server.py              # OSC command handling
│       ├── diffusion_engine.py        # StreamDiffusion integration
│       ├── spout_handler.py           # Spout I/O
│       └── utils.py                   # Helper functions
├── main.py                            # Entry point (handles path resolution)
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── LICENSE                            # MIT License
└── README.md                          # This file
```

**Note:** The service dynamically imports `utils/wrapper.py` from the StreamDiffusion repository at runtime (Apache 2.0 license). The path is automatically resolved—no manual setup needed.

## Dependencies & Credits

This project builds upon:
- **[StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)** - Real-time diffusion pipeline (Apache 2.0 License)
- **[Spout](https://spout.zeal.co/)** - GPU texture sharing for Windows
- **[python-osc](https://github.com/attwad/python-osc)** - OSC protocol implementation

## Compatible Frameworks

Open source creative coding frameworks with Spout support:
- **[openFrameworks](https://openframeworks.cc/)** - C++ creative toolkit, Spout via [ofxSpout](https://github.com/elliotwoods/ofxSpout)
- **[Cinder](https://libcinder.org/)** - C++ library for creative coding, Spout via [Cinder-Spout2](https://github.com/leadedge/Cinder-Spout2)
- **[Processing](https://processing.org/)** - Creative coding in Java, Spout via [Spout for Processing](https://github.com/leadedge/SpoutProcessing)

Commercial applications:
- **[TouchDesigner](https://derivative.ca/)** - Node-based visual programming for real-time projects

## License

MIT License

## Author

[Alex Olwal](https://github.com/olwal)

## Contributing

Issues and pull requests welcome! This is an early-stage project, so feedback is appreciated.
