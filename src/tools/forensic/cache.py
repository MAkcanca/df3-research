"""
Tool output caching for forensic tools.

This module provides caching for deterministic tool outputs to avoid
re-running expensive operations (e.g., GPU inference, image processing)
when the same tool is called with the same inputs.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ToolCache:
    """
    Cache for forensic tool outputs.
    
    Cache keys are based on:
    - Tool name
    - Image file hash (SHA256) - ensures same content = same cache key
    - Tool-specific parameters (e.g., quality for ELA)
    """
    
    def __init__(self, cache_dir: Optional[str] = None, enabled: bool = True):
        """
        Initialize the tool cache.
        
        Args:
            cache_dir: Directory to store cache files. Defaults to .tool_cache in project root.
            enabled: Whether caching is enabled. If False, all cache operations are no-ops.
        """
        self.enabled = enabled
        if not enabled:
            self.cache_dir = None
            return
            
        if cache_dir is None:
            # Default to .tool_cache in project root (parent of src/)
            project_root = Path(__file__).parent.parent.parent.parent
            cache_dir = str(project_root / ".tool_cache")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_image_hash(self, image_path: str) -> str:
        """Get SHA256 hash of image file content, or empty string if file doesn't exist."""
        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash
        except OSError:
            return ""
    
    def _get_image_mtime(self, image_path: str) -> float:
        """Get image modification time, or 0 if file doesn't exist."""
        try:
            return os.path.getmtime(image_path)
        except OSError:
            return 0.0
    
    def _make_cache_key(
        self,
        tool_name: str,
        image_path: str,
        params: Optional[Dict[str, Any]] = None,
        image_hash: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Generate a cache key and filename.
        
        Returns:
            (cache_key_hash, cache_filename)
        """
        # Normalize image path to absolute (for reference, but hash is the key)
        abs_path = str(Path(image_path).resolve())
        
        # Get image hash if not provided
        if image_hash is None:
            image_hash = self._get_image_hash(abs_path)
        
        # Build key components
        key_parts = {
            "tool": tool_name,
            "image_hash": image_hash,
        }
        
        # Add tool-specific parameters (sorted for consistency)
        if params:
            # Filter out None values and sort for consistent hashing
            clean_params = {k: v for k, v in sorted(params.items()) if v is not None}
            key_parts["params"] = clean_params
        
        # Create deterministic JSON string
        key_json = json.dumps(key_parts, sort_keys=True, separators=(',', ':'))
        
        # Hash to create short filename
        key_hash = hashlib.sha256(key_json.encode('utf-8')).hexdigest()[:16]
        
        # Create filename: tool_name_hash.json
        cache_filename = f"{tool_name}_{key_hash}.json"
        
        return key_json, cache_filename
    
    def get(
        self,
        tool_name: str,
        image_path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Get cached tool output if available.
        
        Args:
            tool_name: Name of the tool (e.g., "perform_trufor")
            image_path: Path to the image file
            params: Tool-specific parameters dict
        
        Returns:
            Cached output string, or None if not found/invalid
        """
        if not self.enabled or self.cache_dir is None:
            return None
        
        try:
            image_hash = self._get_image_hash(image_path)
            _, cache_filename = self._make_cache_key(tool_name, image_path, params, image_hash)
            cache_path = self.cache_dir / cache_filename
            
            if not cache_path.exists():
                return None
            
            # Load cache entry
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)
            
            # Verify cache entry is still valid by checking hash
            stored_hash = cache_entry.get("image_hash", "")
            if stored_hash != image_hash:
                # Image content has changed, cache is invalid
                return None
            
            # Return cached output
            return cache_entry.get("output")
        
        except Exception as e:
            # On any error, return None (cache miss)
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Cache get failed for {tool_name}: {e}")
            return None
    
    def set(
        self,
        tool_name: str,
        image_path: str,
        output: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store tool output in cache.
        
        Args:
            tool_name: Name of the tool
            image_path: Path to the image file
            output: Tool output string to cache
            params: Tool-specific parameters dict
        """
        if not self.enabled or self.cache_dir is None:
            return
        
        try:
            image_hash = self._get_image_hash(image_path)
            _, cache_filename = self._make_cache_key(tool_name, image_path, params, image_hash)
            cache_path = self.cache_dir / cache_filename
            
            # Create cache entry
            cache_entry = {
                "tool_name": tool_name,
                "image_path": str(Path(image_path).resolve()),
                "image_hash": image_hash,
                "params": params or {},
                "output": output,
                "cached_at": time.time(),
            }
            
            # Write atomically (write to temp file, then rename)
            temp_path = cache_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2)
            temp_path.replace(cache_path)
        
        except Exception as e:
            # On error, just log and continue (don't fail tool execution)
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Cache set failed for {tool_name}: {e}")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        if not self.enabled or self.cache_dir is None:
            return
        
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Cache clear failed: {e}")
    
    def get_vision_output(
        self,
        vision_model: str,
        image_path: str,
        cache_tag: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached vision model output if available.
        
        Args:
            vision_model: Name of the vision model (e.g., "gpt-5.1", "gpt-5-mini")
            image_path: Path to the image file
            cache_tag: Optional extra cache discriminator (e.g., prompt/version hash).
        
        Returns:
            Cached output dictionary, or None if not found/invalid
        """
        if not self.enabled or self.cache_dir is None:
            return None
        
        try:
            image_hash = self._get_image_hash(image_path)
            # Use "vision" as the tool name for vision model outputs
            params: Dict[str, Any] = {"vision_model": vision_model}
            if cache_tag:
                params["cache_tag"] = str(cache_tag)
            _, cache_filename = self._make_cache_key("vision", image_path, params, image_hash)
            cache_path = self.cache_dir / cache_filename
            
            if not cache_path.exists():
                return None
            
            # Load cache entry
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)
            
            # Verify cache entry is still valid by checking hash
            stored_hash = cache_entry.get("image_hash", "")
            if stored_hash != image_hash:
                # Image content has changed, cache is invalid
                return None
            
            # Return cached output (should be a dict for vision outputs)
            output = cache_entry.get("output")
            if isinstance(output, dict):
                return output
            # Handle legacy string format (shouldn't happen, but be safe)
            if isinstance(output, str):
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    return None
            return None
        
        except Exception as e:
            # On any error, return None (cache miss)
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Vision cache get failed for {vision_model}: {e}")
            return None
    
    def set_vision_output(
        self,
        vision_model: str,
        image_path: str,
        output: Dict[str, Any],
        cache_tag: Optional[str] = None,
    ) -> None:
        """
        Store vision model output in cache.
        
        Args:
            vision_model: Name of the vision model
            image_path: Path to the image file
            output: Vision output dictionary to cache
            cache_tag: Optional extra cache discriminator (e.g., prompt/version hash).
        """
        if not self.enabled or self.cache_dir is None:
            return
        
        try:
            image_hash = self._get_image_hash(image_path)
            # Use "vision" as the tool name for vision model outputs
            params: Dict[str, Any] = {"vision_model": vision_model}
            if cache_tag:
                params["cache_tag"] = str(cache_tag)
            _, cache_filename = self._make_cache_key("vision", image_path, params, image_hash)
            cache_path = self.cache_dir / cache_filename
            
            # Create cache entry
            cache_entry = {
                "tool_name": "vision",
                "vision_model": vision_model,
                "cache_tag": cache_tag,
                "image_path": str(Path(image_path).resolve()),
                "image_hash": image_hash,
                "params": params,
                "output": output,  # Store as dict for vision outputs
                "cached_at": time.time(),
            }
            
            # Write atomically (write to temp file, then rename)
            temp_path = cache_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2)
            temp_path.replace(cache_path)
        
        except Exception as e:
            # On error, just log and continue (don't fail vision execution)
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Vision cache set failed for {vision_model}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or self.cache_dir is None:
            return {"enabled": False}
        
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            return {
                "enabled": True,
                "cache_dir": str(self.cache_dir),
                "entry_count": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Cache stats failed: {e}")
            return {"enabled": True, "error": str(e)}


# Global cache instance (can be configured)
_global_cache: Optional[ToolCache] = None


def get_cache() -> ToolCache:
    """Get the global tool cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ToolCache()
    return _global_cache


def set_cache(cache: ToolCache) -> None:
    """Set the global tool cache instance."""
    global _global_cache
    _global_cache = cache

