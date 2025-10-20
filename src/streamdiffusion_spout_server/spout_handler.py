"""
Spout sender and receiver classes for image sharing
"""
import numpy as np
import SpoutGL
import array
from OpenGL import GL
from PIL import Image

class SpoutReceiver:
    """Class for receiving images via Spout"""

    def __init__(self, name, width, height):
        """
        Initialize Spout receiver.

        Args:
            name: Spout receiver name
            width: Initial image width
            height: Initial image height
        """
        self.name = name
        self.width = width
        self.height = height

        # Initialize Spout receiver
        self.receiver = SpoutGL.SpoutReceiver()
        success = self.receiver.setReceiverName(name)

        from . import config
        if not success and config.verbose >= 2:
            print(f"Note: Spout receiver name not pre-set (will auto-detect sender)")

        # Initialize the buffer for image receiving
        # Buffer will be recreated when sender dimensions are updated
        self.buffer = None

        if config.verbose >= 2:
            print(f"Spout receiver ready for '{name}'")
        
    def receive_frame(self):
        """
        Receive a frame from Spout.

        Returns:
            PIL Image containing the image, or None if no new frame
        """
        # Receive image into buffer
        result = self.receiver.receiveImage(self.buffer, GL.GL_RGBA, False, 0)

        # Check if sender dimensions have been updated
        if self.receiver.isUpdated() and self.receiver.isFrameNew():
            self.width = self.receiver.getSenderWidth()
            self.height = self.receiver.getSenderHeight()
            # Create a new buffer with the updated dimensions
            buffer_size = self.width * self.height * 4
            self.buffer = array.array('B', bytes(buffer_size))
            from . import config
            if config.verbose >= 2:
                print(f"Spout input detected: {self.width}x{self.height}")

            # First time we detect a sender, we need to get the image again
            if result:
                result = self.receiver.receiveImage(self.buffer, GL.GL_RGBA, False, 0)

        # If we have a valid buffer and received something
        if self.buffer and result and not SpoutGL.helpers.isBufferEmpty(self.buffer):
            image = Image.frombuffer('RGBA', (self.width, self.height), self.buffer, 'raw', 'RGBA', 0, 1)
            return image

        return None
        
    def restart(self):
        """Restart the Spout receiver connection"""
        from . import config
        if config.verbose >= 1:
            print(f"SpoutReceiver restarting...", end='')

        # Release existing receiver
        self.receiver.releaseReceiver()

        # Reinitialize receiver
        self.receiver = SpoutGL.SpoutReceiver()
        success = self.receiver.setReceiverName(self.name)
        if not success:
            if config.verbose >= 1:
                print(f"Warning: Could not set receiver name to '{self.name}'")

        # Reset buffer
        self.buffer = None

        if config.verbose >= 1:
            print("[OK]")

        return success

    def close(self):
        """Clean up resources"""
        from . import config
        if config.verbose >= 1:
            print(f"SpoutReceiver closing...", end='')
        self.receiver.releaseReceiver()
        if config.verbose >= 1:
            print("[OK]")


class SpoutSender:
    """Class for sending images via Spout"""

    def __init__(self, name, width, height):
        """
        Initialize Spout sender.

        Args:
            name: Spout sender name
            width: Image width
            height: Image height
        """
        self.name = name
        self.width = width
        self.height = height

        # Initialize Spout sender
        self.sender = SpoutGL.SpoutSender()
        self.sender.setSenderName(name)

        from . import config
        if config.verbose >= 2:
            print(f"Spout sender ready as '{name}'")
        
    def send_frame(self, image):
        """
        Send a frame via Spout.

        Args:
            image: PIL Image to send (RGB or RGBA)

        Returns:
            True if successful, False otherwise
        """
        if image.mode == 'RGBA':
            # If already RGBA, just get the raw bytes
            pixels = image.tobytes()
        elif image.mode == 'RGB':
            rgb_array = np.array(image)
            # Create RGBA array with 255 alpha
            rgba_array = np.ones((self.height, self.width, 4), dtype=np.uint8) * 255
            # Copy RGB data
            rgba_array[:, :, 0:3] = rgb_array
            pixels = rgba_array.tobytes()

        width, height = image.size

        result = self.sender.sendImage(pixels, width, height, GL.GL_RGBA, False, 0)

        return result
    
    def restart(self):
        """Restart the Spout sender connection"""
        from . import config
        if config.verbose >= 1:
            print(f"SpoutSender restarting...", end='')

        # Release existing sender
        self.sender.releaseSender()

        # Reinitialize sender
        self.sender = SpoutGL.SpoutSender()
        self.sender.setSenderName(self.name)

        if config.verbose >= 1:
            print("[OK]")

    def close(self):
        """Clean up resources"""
        from . import config
        if config.verbose >= 1:
            print(f"SpoutSender [{self.name}] closing...", end='')
        self.sender.releaseSender()
        if config.verbose >= 1:
            print("[OK]")
