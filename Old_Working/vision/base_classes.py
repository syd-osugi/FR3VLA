"""
Base Classes for Interchangeability
------------------------------------
This file uses Python's ABC (Abstract Base Class) module.

WHY DO WE NEED THIS?
If you swap Qwen for a different LLM (like Claude or a local HuggingFace model), that 
new model might have a completely different API. However, your main.py and tools.py 
expect specific functions like `send_message_with_tools()` to exist.

By defining a BaseLLM class here, we create a strict CONTRACT. Any future LLM file 
MUST have these exact functions, or Python will refuse to run it. This guarantees that 
main.py will never break when you swap the AI brain.
"""

from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    The contract that ALL LLM interfaces must follow.
    """
    
    @abstractmethod
    def get_text(self):
        """Must implement a way to get user input."""
        pass

    @abstractmethod
    def send_message_with_tools(self, d435_cam, d405_cam):
        """Must implement the tool-calling loop."""
        pass

    @abstractmethod
    def prune_image_history(self):
        """Must implement a way to clear memory."""
        pass

    @abstractmethod
    def print_message(self):
        """Must implement a way to display the final text."""
        pass


class BaseDetector(ABC):
    """
    The contract that ALL vision models (YOLO, SAM, etc.) must follow.
    """
    
    @abstractmethod
    def detect(self, image, classes_to_find=None):
        """Must take an image and return a list of detected objects."""
        pass