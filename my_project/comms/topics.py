"""MQTT topic definitions and helpers"""


class MQTTTopics:
    """MQTT topic definitions"""
    
    # Data topics
    PICOAMMETER = "picoammeter/current"
    STAGE_POSITION = "microscope/stage/position"
    
    # Command topics
    STAGE_COMMAND = "microscope/stage/command"
    STAGE_RESULT = "microscope/stage/result"
    
    # Status topics (future expansion)
    SYSTEM_STATUS = "microscope/system/status"
    ERROR_REPORT = "microscope/system/error"
    
    @staticmethod
    def is_data_topic(topic: str) -> bool:
        """Check if topic is a data topic"""
        return topic in [MQTTTopics.PICOAMMETER, MQTTTopics.STAGE_POSITION]
    
    @staticmethod
    def is_command_topic(topic: str) -> bool:
        """Check if topic is a command topic"""
        return topic in [MQTTTopics.STAGE_COMMAND, MQTTTopics.STAGE_RESULT]
