"""
OSC server implementation for receiving commands
"""
import socket
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from . import config

def process_set_prompt(address, *args):
    """
    OSC handler for setting prompts.

    Args:
        address: OSC address
        *args: OSC arguments (prompt and optional negative prompt)
    """
    if len(args) >= 1:
        prompt = str(args[0])
        config.current_prompt = prompt
        if config.verbose >= 2:
            print(f"Prompt: {prompt[:40]}...")

    if len(args) >= 2:
        config.current_negative_prompt = str(args[1])
        if config.verbose >= 3:
            print(f"Negative prompt: {config.current_negative_prompt}")

    # Put the new prompt in the queue
    config.prompt_queue.put((config.current_prompt, config.current_negative_prompt))

def process_trigger(address, *args):
    """
    OSC handler for triggering image generation.

    Args:
        address: OSC address
        *args: OSC arguments (unused)
    """
    if config.verbose >= 2:
        print("Generation triggered")
    config.trigger_event.set()

def process_continuous_start(address, *args):
    """
    OSC handler for starting continuous generation.

    Args:
        address: OSC address
        *args: OSC arguments (unused)
    """
    # Only print if state is changing
    if not config.start_event.is_set() and config.verbose >= 2:
        print("Continuous started")
    config.start_event.set()

def process_continuous_stop(address, *args):
    """
    OSC handler for stopping continuous generation.

    Args:
        address: OSC address
        *args: OSC arguments (unused)
    """
    # Only print if state is changing
    if config.start_event.is_set() and config.verbose >= 2:
        print("Continuous stopped")
    config.stop_event.set()

def process_spout_start(address, *args):
    """
    OSC handler for enabling Spout output.

    Args:
        address: OSC address
        *args: OSC arguments (unused)
    """
    # Only print if state is changing
    if not config.spout_send_event.is_set() and config.verbose >= 2:
        print("Spout started")
    config.spout_send_event.set()

def process_spout_stop(address, *args):
    """
    OSC handler for disabling Spout output.

    Args:
        address: OSC address
        *args: OSC arguments (unused)
    """
    # Only print if state is changing
    if config.spout_send_event.is_set() and config.verbose >= 2:
        print("Spout stopped")
    config.spout_send_event.clear()

def process_verbose_set(address, *args):
    """
    OSC handler for setting verbose level.

    Args:
        address: OSC address
        *args: OSC arguments (verbose level 0-3)
    """
    if len(args) >= 1:
        level = int(args[0])
        if 0 <= level <= 3:
            config.verbose = level
            print(f"Verbose level set to: {level}")
        else:
            print(f"Invalid verbose level: {level} (must be 0-3)")
    else:
        print(f"Current verbose level: {config.verbose}")

def process_verbose_toggle(address, *args):
    """
    OSC handler for cycling through verbose levels 0-3.

    Args:
        address: OSC address
        *args: OSC arguments (unused)
    """
    config.verbose = (config.verbose + 1) % 4
    print(f"Verbose level: {config.verbose}")

def process_verbose_on(address, *args):
    """
    OSC handler for enabling verbose mode (level 2).

    Args:
        address: OSC address
        *args: OSC arguments (unused)
    """
    config.verbose = 2
    print("Verbose level: 2")

def process_verbose_off(address, *args):
    """
    OSC handler for disabling verbose mode (level 0).

    Args:
        address: OSC address
        *args: OSC arguments (unused)
    """
    config.verbose = 0
    print("Verbose level: 0 (quiet)")

def process_spout_restart(address, *args):
    """
    OSC handler for restarting Spout connections.

    Args:
        address: OSC address
        *args: OSC arguments (unused)
    """
    if config.verbose >= 1:
        print("Spout restart requested")
    config.spout_restart_event.set()


def start_osc_server(ip, port):
    """
    Start the OSC server thread.
   
    Args:
        ip: IP address to listen on
        port: Port to listen on
    """
    dispatcher = Dispatcher()
    dispatcher.map("/prompt", process_set_prompt)
    dispatcher.map("/trigger", process_trigger)
    dispatcher.map("/t", process_trigger)
    dispatcher.map("/start", process_continuous_start)
    dispatcher.map("/stop", process_continuous_stop)
    dispatcher.map("/s", process_continuous_start)
    dispatcher.map("/S", process_continuous_stop)
    dispatcher.map("/p", process_spout_start)    
    dispatcher.map("/P", process_spout_stop)
    dispatcher.map("/verbose", process_verbose_set)
    dispatcher.map("/v", process_verbose_toggle)
    dispatcher.map("/von", process_verbose_on)
    dispatcher.map("/voff", process_verbose_off)
    dispatcher.map("/x", process_spout_restart)        
   
    server = BlockingOSCUDPServer((ip, port), dispatcher)
    
    # Set a timeout on the server socket
    server.socket.settimeout(0.5)  # Check exit flag every 0.5 seconds
    
    if config.verbose >= 1:
        print("--------------------")
        print(f"OSC server listening on {ip}:{port}")
        print("--------------------")
   
    # Keep running until exit flag is set
    while not config.exit_flag.is_set():
        try:
            server.handle_request()
        except socket.timeout:
            # This is expected - just allows us to check the flag
            pass
        except Exception as e:
            print(f"Error in OSC server: {e}")
            break