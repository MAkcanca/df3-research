"""
Code execution tool for the forensic agent.

This allows the agent to execute Python code dynamically, similar to ChatGPT's code interpreter.
Useful for custom image analysis, zooming, cropping, statistical analysis, etc.
"""

import io
import os
import shutil
import traceback
from typing import Dict, Any, Optional
from pathlib import Path
import base64
import json

# Artifacts directory for code execution outputs
# This is where any generated files (plots, images, etc.) should be saved
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts"


def get_artifacts_dir() -> Path:
    """Get the artifacts directory, creating it if it doesn't exist."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def clean_artifacts_dir() -> None:
    """Remove all files from the artifacts directory."""
    if ARTIFACTS_DIR.exists():
        shutil.rmtree(ARTIFACTS_DIR)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None


# Whitelist of allowed modules for import
ALLOWED_MODULES = {
    'numpy', 'np',
    'PIL', 'PIL.Image', 'PIL.ImageFilter', 'PIL.ImageEnhance', 'PIL.ImageOps',
    'cv2',
    'scipy', 'scipy.ndimage', 'scipy.fft', 'scipy.signal', 'scipy.stats',
    'math', 'statistics',
    'collections', 'itertools', 'functools',
    'json', 'base64', 'io', 'struct',
    'pathlib',
    'matplotlib', 'matplotlib.pyplot',
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricted import that only allows whitelisted modules."""
    import builtins
    # Check if the base module is allowed
    base_module = name.split('.')[0]
    if name in ALLOWED_MODULES or base_module in ALLOWED_MODULES:
        return builtins.__import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Module '{name}' is not allowed. Allowed modules: {sorted(ALLOWED_MODULES)}")


def execute_python_code(code: str, image_path: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Execute Python code in a sandboxed environment with image processing capabilities.
    
    Args:
        code: Python code to execute
        image_path: Optional path to the current image being analyzed
        context: Optional dictionary of variables to make available to the code
        
    Returns:
        String result of code execution (stdout + return value if any)
    """
    # Capture output WITHOUT mutating global sys.stdout/sys.stderr (thread-safe).
    # We capture print() by providing a sandboxed print implementation that writes to these buffers.
    captured_output = io.StringIO()
    captured_error = io.StringIO()

    def _sandbox_print(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False) -> None:
        """
        A minimal print() replacement that writes to per-call buffers.

        - Defaults to captured_output
        - Supports file=... for advanced usage (e.g., file=captured_error)
        - Never writes to real process stdout/stderr
        """
        target = captured_output if file is None else file
        try:
            text = sep.join(str(a) for a in args) + end
            target.write(text)
            if flush and hasattr(target, "flush"):
                target.flush()
        except Exception as e:
            # Best-effort: record print failures in captured_error, but never raise.
            try:
                captured_error.write(f"[print error] {e}\n")
            except Exception:
                pass
    
    try:
        # Create execution namespace with safe builtins
        safe_builtins = {
            # Basic types and constructors
            'abs': abs, 'all': all, 'any': any, 'bool': bool, 'bytes': bytes,
            'bytearray': bytearray, 'dict': dict, 'float': float, 'frozenset': frozenset,
            'int': int, 'list': list, 'set': set, 'str': str, 'tuple': tuple,
            # Iteration and sequences
            'enumerate': enumerate, 'filter': filter, 'iter': iter, 'len': len,
            'map': map, 'max': max, 'min': min, 'next': next, 'range': range,
            'reversed': reversed, 'slice': slice, 'sorted': sorted, 'sum': sum, 'zip': zip,
            # Type checking and introspection
            'callable': callable, 'getattr': getattr, 'hasattr': hasattr,
            'isinstance': isinstance, 'issubclass': issubclass, 'type': type,
            # Math and representation
            'divmod': divmod, 'pow': pow, 'repr': repr, 'round': round,
            # I/O (print only, no file open)
            'print': _sandbox_print,
            # Import (restricted to whitelist)
            '__import__': _safe_import,
            # Exceptions (for proper error handling in code)
            'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
            'KeyError': KeyError, 'IndexError': IndexError, 'RuntimeError': RuntimeError,
            'StopIteration': StopIteration, 'ZeroDivisionError': ZeroDivisionError,
        }
        
        namespace = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
        }
        
        # Pre-import common libraries for convenience
        if np is not None:
            namespace['np'] = np
            namespace['numpy'] = np
        if Image is not None:
            namespace['Image'] = Image
            try:
                import PIL
                namespace['PIL'] = PIL
            except ImportError:
                pass
        namespace['Path'] = Path
        namespace['base64'] = base64
        namespace['json'] = json
        namespace['io'] = io
        try:
            import math
            namespace['math'] = math
        except ImportError:
            pass
        
        # Add artifacts directory for saving outputs
        artifacts_dir = get_artifacts_dir()
        namespace['artifacts_dir'] = artifacts_dir
        namespace['ARTIFACTS_DIR'] = artifacts_dir
        
        # Add image path if provided
        if image_path:
            namespace['image_path'] = image_path
            namespace['current_image_path'] = image_path
            
            # Load image if PIL is available
            if Image:
                try:
                    img = Image.open(image_path)
                    namespace['image'] = img
                    namespace['img'] = img
                    # Also provide as numpy array if numpy is available
                    if np:
                        img_array = np.array(img)
                        namespace['image_array'] = img_array
                        namespace['img_array'] = img_array
                except Exception as e:
                    namespace['image_load_error'] = str(e)
        
        # Add any context variables
        if context:
            namespace.update(context)
        
        # Execute the code
        exec(code, namespace)
        
        # Check if there's a return value
        result_value = namespace.get('result', None)
        
        # Get captured output
        stdout_output = captured_output.getvalue()
        stderr_output = captured_error.getvalue()
        
        # Combine outputs
        output_parts = []
        if stdout_output:
            output_parts.append(stdout_output)
        if stderr_output:
            output_parts.append(f"STDERR:\n{stderr_output}")
        if result_value is not None:
            output_parts.append(f"\nReturn value: {result_value}")
        
        return '\n'.join(output_parts) if output_parts else "Code executed successfully (no output)"
        
    except Exception:
        error_traceback = traceback.format_exc()
        # Include captured output to aid debugging (thread-safe, no global stdout capture)
        stdout_output = captured_output.getvalue()
        stderr_output = captured_error.getvalue()
        out = []
        if stdout_output:
            out.append(stdout_output)
        if stderr_output:
            out.append(f"STDERR:\n{stderr_output}")
        out.append(f"Error executing code:\n{error_traceback}")
        return "\n".join(out)


def run_code_interpreter(input_str: str) -> str:
    """
    LangChain tool wrapper for code execution.
    
    Input format: JSON string with 'code' and optionally 'image_path'
    Example: '{"code": "print(image.size)", "image_path": "path/to/image.jpg"}'
    Or simple string: just the code (will try to extract image_path from context if available)
    
    The agent should include the image_path in the JSON if available from the analysis context.
    """
    try:
        # Try to parse as JSON first
        try:
            params = json.loads(input_str)
            code = params.get('code', '')
            image_path = params.get('image_path')
            context = params.get('context', {})
        except (json.JSONDecodeError, AttributeError):
            # If not JSON, treat as plain code string
            code = input_str
            image_path = None
            context = {}
            
            # Try to extract image_path from the code string if it mentions it
            # This is a fallback - ideally the agent should pass it in JSON
            import re
            path_match = re.search(r'image_path\s*=\s*["\']([^"\']+)["\']', code)
            if path_match:
                image_path = path_match.group(1)
        
        if not code.strip():
            return "Error: No code provided. Provide Python code to execute.\n" \
                   "Format: {\"code\": \"your_python_code_here\", \"image_path\": \"path/to/image.jpg\"}\n" \
                   "Or: just the Python code as a string (image_path should be in agent context)."
        
        # If no image_path provided, try to find it from common patterns in code
        if not image_path:
            import re
            # Look for image_path variable assignment or usage
            path_patterns = [
                r'image_path\s*=\s*["\']([^"\']+)["\']',
                r'["\']([^"\']+\.(jpg|jpeg|png|gif|bmp))["\']',
            ]
            for pattern in path_patterns:
                match = re.search(pattern, code, re.IGNORECASE)
                if match:
                    potential_path = match.group(1)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
        
        # Execute the code
        result = execute_python_code(code, image_path=image_path, context=context)
        return result
        
    except Exception as e:
        return f"Error in code interpreter: {str(e)}\n{traceback.format_exc()}"

