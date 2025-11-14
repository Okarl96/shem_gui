"""
Command Sender for Microscope Stage Control

Formats and sends movement/stop commands to the microscope stage via MQTT.
"""

from typing import Optional, List, Tuple
from .mqtt_client import MQTTClient
import time


class CommandSender:
    """
    Sends movement commands to microscope stage.
    
    Formats commands in the protocol expected by the stage controller
    and handles MQTT communication.
    """
    
    def __init__(self, mqtt_client: MQTTClient):
        """
        Initialize command sender.
        
        Args:
            mqtt_client: MQTTClient instance for communication
        """
        self.mqtt = mqtt_client
        
        # Valid axis names
        self.valid_axes = {'X', 'Y', 'Z', 'R'}
    
    # --- Single axis operations ---
    
    def move_axis(self, axis: str, position: float) -> bool:
        """
        Move single axis to absolute position.
        
        Args:
            axis: Axis name ('X', 'Y', 'Z', or 'R')
            position: Target position in nm (or udeg for R axis)
            
        Returns:
            True if command sent successfully
            
        Example:
            commander.move_axis('X', 5000)  # Move X to 5000 nm
            → MOVE/X/5000
        """
        if not self._validate_axis(axis):
            return False
        
        command = self._format_move_command(axis, position)
        success = self.mqtt.send_command(command)
        
        if success:
            print(f"→ {command}")
        else:
            print(f"Failed to send: {command}")
        
        return success
    
    def stop_axis(self, axis: str) -> bool:
        """
        Stop single axis.
        
        Args:
            axis: Axis name ('X', 'Y', 'Z', or 'R')
            
        Returns:
            True if command sent successfully
            
        Example:
            commander.stop_axis('X')
            ⏹ STOP/X
        """
        if not self._validate_axis(axis):
            return False
        
        command = self._format_stop_command(axis)
        success = self.mqtt.send_command(command)
        
        if success:
            print(f"{command}")
        else:
            print(f"Failed to send: {command}")
        
        return success
    
    # --- Multi-axis operations ---
    
    def move_to(self, 
                X: Optional[float] = None,
                Y: Optional[float] = None,
                Z: Optional[float] = None,
                R: Optional[float] = None) -> bool:
        """
        Move multiple axes to absolute positions simultaneously.
        Only moves axes with non-None values.
        
        Args:
            x: X position in nm (None = don't move this axis)
            y: Y position in nm (None = don't move this axis)
            z: Z position in nm (None = don't move this axis)
            r: R position in udeg (None = don't move this axis)
            
        Returns:
            True if all commands sent successfully
            
        Example:
            commander.move_to(x=5000, y=10000, z=2000)
            → MOVE/X/5000
            → MOVE/Y/10000
            → MOVE/Z/2000
        """
        positions = {'X': X, 'Y': Y, 'Z': Z, 'R': R}
        success = True
        
        for axis, pos in positions.items():
            if pos is not None:
                if not self.move_axis(axis, pos):
                    success = False
        
        return success
    
    def stop_all(self) -> bool:
        """
        Stop all axes.
        
        Returns:
            True if all commands sent successfully
        """
        success = True
        for axis in self.valid_axes:
            if not self.stop_axis(axis):
                success = False
        return success

    # --- Batch operations ---

    def send_batch_commands(self, commands: List[Tuple[str, float]]) -> bool:
        """
        Send multiple movement commands in sequence.

        Args:
            commands: List of (axis, position) tuples

        Returns:
            True if all commands sent successfully

        Example:
            commander.send_batch_commands([
            ...     ('X', 5000),
            ...     ('Y', 10000),
            ...     ('Z', 2000)
            ... ])
        """
        success = True
        for i, (axis, position) in enumerate(commands):
            if not self.move_axis(axis, position):
                success = False
            # Add small pause between commands (except after last one)
            if i < len(commands) - 1:
                time.sleep(0.1)
        return success
    
    # --- Internal helper methods ---
    
    def _validate_axis(self, axis: str) -> bool:
        """
        Validate axis name.
        
        Args:
            axis: Axis name to validate
            
        Returns:
            True if valid, False otherwise
        """
        axis_upper = axis.upper()
        if axis_upper not in self.valid_axes:
            print(f"✗ Invalid axis '{axis}'. Must be one of: {self.valid_axes}")
            return False
        return True
    
    def _format_move_command(self, axis: str, position: float) -> str:
        """
        Format movement command string.
        
        Protocol: MOVE/<AXIS>/<POSITION>
        Position is rounded to nearest integer.
        
        Args:
            axis: Axis name (X, Y, Z, R)
            position: Target position
            
        Returns:
            Formatted command string
        """
        return f"MOVE/{axis.upper()}/{position:.0f}"
    
    def _format_stop_command(self, axis: str) -> str:
        """
        Format stop command string.
        
        Protocol: STOP/<AXIS>
        
        Args:
            axis: Axis name (X, Y, Z, R)
            
        Returns:
            Formatted command string
        """
        return f"STOP/{axis.upper()}"
    
    # --- Status and utility methods ---
    
    def is_ready(self) -> bool:
        """
        Check if command sender is ready to send commands.
        
        Returns:
            True if MQTT connection is active
        """
        return self.mqtt.is_connected()
    
    def get_valid_axes(self) -> set:
        """
        Get set of valid axis names.
        
        Returns:
            Set of valid axis names
        """
        return self.valid_axes.copy()
