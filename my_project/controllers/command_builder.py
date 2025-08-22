"""Command builder for stage control"""


class CommandBuilder:
    """Build MQTT commands for stage control"""
    
    @staticmethod
    def move_x(position: float) -> str:
        """Build X axis move command"""
        return f"MOVE/X/{position:.0f}"
    
    @staticmethod
    def move_y(position: float) -> str:
        """Build Y axis move command"""
        return f"MOVE/Y/{position:.0f}"
    
    @staticmethod
    def move_z(position: float) -> str:
        """Build Z axis move command"""
        return f"MOVE/Z/{position:.0f}"
    
    @staticmethod
    def move_r(position: float) -> str:
        """Build R axis move command"""
        return f"MOVE/R/{position:.0f}"
    
    @staticmethod
    def stop_x() -> str:
        """Build X axis stop command"""
        return "STOP/X"
    
    @staticmethod
    def stop_y() -> str:
        """Build Y axis stop command"""
        return "STOP/Y"
    
    @staticmethod
    def stop_all() -> str:
        """Build stop all axes command"""
        return "STOP/ALL"
    
    @staticmethod
    def set_velocity(axis: str, velocity: float) -> str:
        """Build velocity setting command"""
        return f"SET/VEL/{axis}/{velocity:.2f}"
    
    @staticmethod
    def get_position(axis: str) -> str:
        """Build position query command"""
        return f"GET/POS/{axis}"
    
    @staticmethod
    def get_status() -> str:
        """Build status query command"""
        return "GET/STATUS"
