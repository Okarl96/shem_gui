"""Plugin loader and manager"""

import importlib
import pkgutil
from pathlib import Path
from typing import List, Dict

from .base import ITabPlugin


class PluginLoader:
    """Load and manage plugins"""
    
    def __init__(self, plugin_dir: str = None):
        self.plugin_dir = plugin_dir or Path(__file__).parent / "builtin"
        self.plugins: Dict[str, ITabPlugin] = {}
        
    def discover_plugins(self) -> List[str]:
        """Discover available plugins"""
        discovered = []
        
        # Load built-in plugins
        builtin_path = Path(__file__).parent / "builtin"
        if builtin_path.exists():
            for module_info in pkgutil.iter_modules([str(builtin_path)]):
                discovered.append(f"builtin.{module_info.name}")
                
        # Load external plugins if directory specified
        if self.plugin_dir and self.plugin_dir.exists():
            for module_info in pkgutil.iter_modules([str(self.plugin_dir)]):
                discovered.append(module_info.name)
                
        return discovered
        
    def load_plugin(self, plugin_name: str) -> ITabPlugin:
        """Load a specific plugin"""
        if plugin_name in self.plugins:
            return self.plugins[plugin_name]
            
        try:
            if plugin_name.startswith("builtin."):
                module_name = f"plugins.{plugin_name}"
            else:
                module_name = plugin_name
                
            module = importlib.import_module(module_name)
            
            # Find the plugin class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    hasattr(attr, 'create_widget') and
                    attr.__name__ != 'ITabPlugin'):
                    
                    plugin = attr()
                    self.plugins[plugin_name] = plugin
                    return plugin
                    
        except Exception as e:
            print(f"Failed to load plugin {plugin_name}: {e}")
            
        return None
        
    def load_all_plugins(self) -> Dict[str, ITabPlugin]:
        """Load all discovered plugins"""
        for plugin_name in self.discover_plugins():
            self.load_plugin(plugin_name)
            
        return self.plugins
