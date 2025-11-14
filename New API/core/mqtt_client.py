"""
MQTT Client for Microscope Communication

Provides basic MQTT connectivity for receiving position/current data
and sending commands to the microscope stage.
"""

import paho.mqtt.client as mqtt
import threading
import time
from typing import Callable, Optional


class MQTTClient:
    """
    Minimal MQTT client for microscope communication.
    
    Handles connection to MQTT broker, subscribes to position and current data,
    and publishes movement commands.
    """
    
    def __init__(self, host: str = "localhost", port: int = 1883):
        """
        Initialize MQTT client.
        
        Args:
            host: MQTT broker hostname/IP
            port: MQTT broker port (default: 1883)
        """
        self.host = host
        self.port = port
        self.connected = False
        self._lock = threading.Lock()
        
        # Default topic configuration
        self.topics = {
            'position': "microscope/stage/position",
            'current': "picoammeter/current",
            'command': "microscope/stage/command",
        }
        
        # Callbacks for data streams
        self.on_position: Optional[Callable[[str], None]] = None
        self.on_current: Optional[Callable[[str], None]] = None
        
        # Create MQTT client with version compatibility
        try:
            # Try new API first (paho-mqtt >= 2.0)
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except (AttributeError, TypeError):
            # Fallback to old API (paho-mqtt < 2.0)
            self.client = mqtt.Client()
        
        # Set internal callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
    
    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to MQTT broker.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            # Attempt connection
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            
            # Wait for connection with timeout
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.connected:
                print(f"Connected to MQTT broker at {self.host}:{self.port}")
                return True
            else:
                print(f"Connection timeout after {timeout}s")
                self.client.loop_stop()
                return False
                
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker and stop background thread."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            with self._lock:
                self.connected = False
            print("Disconnected from MQTT broker")
    
    def send_command(self, command: str) -> bool:
        """
        Send command to microscope stage.
        
        Args:
            command: Command string (e.g., "MOVE/X/5000")
            
        Returns:
            True if command was published successfully
        """
        if not self.is_connected():
            print("Cannot send command: Not connected to broker")
            return False
        
        try:
            result = self.client.publish(self.topics['command'], command)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            print(f"Command send failed: {e}")
            return False
    
    def set_callbacks(self,
                     on_position: Optional[Callable[[str], None]] = None,
                     on_current: Optional[Callable[[str], None]] = None):
        """
        Set callback functions for data streams.
        
        Args:
            on_position: Called when position message received (~6kHz from ECC100)
            on_current: Called when current message received (~40Hz from picoammeter)
        """
        if on_position is not None:
            self.on_position = on_position
        if on_current is not None:
            self.on_current = on_current
    
    def is_connected(self) -> bool:
        """
        Check if connected to MQTT broker.
        
        Returns:
            True if connected, False otherwise
        """
        with self._lock:
            return self.connected
    
    def set_topics(self, position: str = None, current: str = None, command: str = None):
        """
        Override default MQTT topics.
        
        Args:
            position: Position data topic
            current: Current data topic
            command: Command publishing topic
        """
        if position:
            self.topics['position'] = position
        if current:
            self.topics['current'] = current
        if command:
            self.topics['command'] = command
    
    # --- Internal callbacks ---
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Internal callback when connected to broker."""
        if rc == 0:
            # Connection successful
            with self._lock:
                self.connected = True
            
            # Subscribe to data topics (not command topic - we only publish to it)
            client.subscribe(self.topics['position'])
            client.subscribe(self.topics['current'])
            
            print(f"Subscribed to: position, current")
        else:
            print(f"Connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc, properties=None, reason_code=None):
        """Internal callback when disconnected from broker."""
        with self._lock:
            self.connected = False
        
        if rc == 0:
            # Clean disconnect
            pass
        else:
            print(f"Unexpected disconnect (rc={rc})")
    
    def _on_message(self, client, userdata, msg):
        """Internal callback when message received - routes to user callbacks."""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            # Route to appropriate callback
            if topic == self.topics['position'] and self.on_position:
                self.on_position(payload)
            elif topic == self.topics['current'] and self.on_current:
                self.on_current(payload)
                
        except Exception as e:
            # Silent fail to avoid flooding console at high data rates
            pass
    
    def __enter__(self):
        """Context manager entry - auto-connect."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-disconnect."""
        self.disconnect()
        return False
