"""MQTT client implementation"""

import paho.mqtt.client as mqtt
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from .topics import MQTTTopics
from core.exceptions import ConnectionError


class MQTTController(QObject):
    """MQTT communication controller"""
    
    # Signals
    connected = pyqtSignal(bool)
    position_data_received = pyqtSignal(str)
    current_data_received = pyqtSignal(str)
    command_result_received = pyqtSignal(str)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.broker_host = "localhost"
        self.broker_port = 1883
        self.connected_status = False
        self.topics = MQTTTopics()
        self.subscribed_topics = set()  # Track subscriptions
        self.subscription_status = {}   # Track subscription results
        
    def setup_mqtt(self, host: str, port: int):
        """Setup MQTT client"""
        self.broker_host = host
        self.broker_port = port
        
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback with subscription tracking"""
        if rc == 0:
            self.connected_status = True
            self.connected.emit(True)
            
            # Subscribe to inbound topics and track
            topics_to_subscribe = [
                self.topics.PICOAMMETER,
                self.topics.STAGE_POSITION,
                self.topics.STAGE_RESULT
            ]
            
            for topic in topics_to_subscribe:
                result = client.subscribe(topic)
                if result[0] == 0:
                    self.subscribed_topics.add(topic)
                    self.subscription_status[topic] = True
                else:
                    self.subscription_status[topic] = False
                    self.error_occurred.emit(f"Failed to subscribe to {topic}")
            
            self.status_update.emit(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
        else:
            self.connected_status = False
            self.connected.emit(False)
            error_msg = self._get_connection_error_message(rc)
            self.error_occurred.emit(f"MQTT connection failed: {error_msg}")
            
    def verify_subscriptions(self) -> tuple[bool, list]:
        """Verify all required subscriptions are active
        
        Returns:
            (all_subscribed, missing_topics)
        """
        required_topics = [
            self.topics.PICOAMMETER,
            self.topics.STAGE_POSITION
        ]
        
        missing = []
        for topic in required_topics:
            if topic not in self.subscribed_topics:
                missing.append(topic)
                
        return len(missing) == 0, missing
        
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.connected_status = False
        self.connected.emit(False)
        self.subscribed_topics.clear()  # Clear subscription tracking
        self.subscription_status.clear()
        
        if rc != 0:
            self.error_occurred.emit(f"Unexpected disconnection (code: {rc})")
        self.status_update.emit("Disconnected from MQTT broker")
        
    def _on_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            topic = msg.topic
            payload = msg.payload.decode()
            
            if topic == self.topics.PICOAMMETER:
                self.current_data_received.emit(payload)
            elif topic == self.topics.STAGE_POSITION:
                self.position_data_received.emit(payload)
            elif topic == self.topics.STAGE_RESULT:
                self.command_result_received.emit(payload)
                
        except Exception as e:
            self.error_occurred.emit(f"Message processing error: {e}")
            
    @pyqtSlot()
    def connect_mqtt(self):
        """Connect to MQTT broker"""
        if not self.client:
            raise ConnectionError("MQTT client not initialized. Call setup_mqtt first.")
            
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
        except Exception as e:
            self.error_occurred.emit(f"Connection error: {e}")
            raise ConnectionError(f"Failed to connect to MQTT broker: {e}")
            
    @pyqtSlot()
    def disconnect_mqtt(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            
    @pyqtSlot(str)
    def send_command(self, command: str):
        """Send command via MQTT"""
        if not self.client or not self.connected_status:
            self.error_occurred.emit("Cannot send command: Not connected to MQTT broker")
            return
            
        try:
            self.client.publish(self.topics.STAGE_COMMAND, command)
            self.status_update.emit(f"Sent command: {command}")
        except Exception as e:
            self.error_occurred.emit(f"Command send error: {e}")
            
    def _get_connection_error_message(self, rc: int) -> str:
        """Get human-readable connection error message"""
        errors = {
            1: "Incorrect protocol version",
            2: "Invalid client identifier",
            3: "Server unavailable",
            4: "Bad username or password",
            5: "Not authorized"
        }
        return errors.get(rc, f"Unknown error (code: {rc})")
