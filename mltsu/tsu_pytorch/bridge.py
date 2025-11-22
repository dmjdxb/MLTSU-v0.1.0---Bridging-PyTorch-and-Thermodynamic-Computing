"""
Bridge between PyTorch and JAX/NumPy for TSU operations
Handles tensor conversions and device management
"""

import torch
import numpy as np
from typing import Optional, Union, Any

# Try to import JAX - make it optional for non-JAX backends
try:
    import jax
    import jax.numpy as jnp
    import jax.dlpack
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None


def torch_to_jax(tensor: torch.Tensor) -> Any:
    """
    Convert PyTorch tensor to JAX array via DLPack.

    Args:
        tensor: PyTorch tensor

    Returns:
        JAX array

    Raises:
        ImportError: If JAX is not installed
        RuntimeError: If conversion fails
    """
    if not HAS_JAX:
        raise ImportError("JAX is required for torch_to_jax conversion")

    # Ensure tensor is on CPU or CUDA (not MPS)
    if tensor.device.type == "mps":
        tensor = tensor.cpu()

    # Use DLPack for zero-copy conversion when possible
    try:
        # For CUDA tensors, ensure they're on the same device as JAX
        if tensor.is_cuda:
            dlpack = torch.to_dlpack(tensor)
            return jax.dlpack.from_dlpack(dlpack)
        else:
            # For CPU tensors, convert via numpy (more stable)
            return jnp.array(tensor.detach().numpy())
    except Exception as e:
        # Fallback to numpy conversion
        return jnp.array(tensor.detach().cpu().numpy())


def jax_to_torch(
    array: Any,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert JAX array to PyTorch tensor.

    Args:
        array: JAX array
        device: Target PyTorch device
        dtype: Target PyTorch dtype

    Returns:
        PyTorch tensor

    Raises:
        ImportError: If JAX is not installed
    """
    if not HAS_JAX:
        raise ImportError("JAX is required for jax_to_torch conversion")

    # Convert to numpy first (most stable path)
    np_array = np.array(array)

    # Create torch tensor
    tensor = torch.from_numpy(np_array)

    # Convert dtype if specified
    if dtype is not None:
        tensor = tensor.to(dtype)
    elif np_array.dtype == np.float64:
        tensor = tensor.float()  # Default to float32 for efficiency

    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def numpy_to_torch(
    array: np.ndarray,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    Args:
        array: Numpy array
        device: Target PyTorch device
        dtype: Target PyTorch dtype

    Returns:
        PyTorch tensor
    """
    # Create tensor from numpy
    tensor = torch.from_numpy(array.copy())  # Copy to avoid shared memory issues

    # Convert dtype if specified
    if dtype is not None:
        tensor = tensor.to(dtype)
    elif array.dtype == np.float64:
        tensor = tensor.float()  # Default to float32

    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)

    return tensor


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.

    Args:
        tensor: PyTorch tensor

    Returns:
        Numpy array
    """
    return tensor.detach().cpu().numpy()


class TensorConverter:
    """
    Context manager for automatic tensor conversion between frameworks.
    """

    def __init__(self, target_framework: str = "jax"):
        """
        Initialize converter.

        Args:
            target_framework: Target framework ('jax', 'numpy', 'torch')
        """
        self.target_framework = target_framework
        self.original_tensors = {}

    def convert_inputs(self, *args, **kwargs):
        """
        Convert input tensors to target framework.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Converted arguments
        """
        converted_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                self.original_tensors[f"arg_{i}"] = arg
                if self.target_framework == "jax" and HAS_JAX:
                    converted_args.append(torch_to_jax(arg))
                elif self.target_framework == "numpy":
                    converted_args.append(torch_to_numpy(arg))
                else:
                    converted_args.append(arg)
            else:
                converted_args.append(arg)

        converted_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                self.original_tensors[f"kwarg_{key}"] = value
                if self.target_framework == "jax" and HAS_JAX:
                    converted_kwargs[key] = torch_to_jax(value)
                elif self.target_framework == "numpy":
                    converted_kwargs[key] = torch_to_numpy(value)
                else:
                    converted_kwargs[key] = value
            else:
                converted_kwargs[key] = value

        return converted_args, converted_kwargs

    def convert_outputs(self, outputs, device=None):
        """
        Convert outputs back to PyTorch tensors.

        Args:
            outputs: Output from target framework
            device: Target device for PyTorch tensors

        Returns:
            PyTorch tensors
        """
        if self.target_framework == "jax" and HAS_JAX:
            if isinstance(outputs, (list, tuple)):
                return type(outputs)(
                    jax_to_torch(o, device) if hasattr(o, "shape") else o
                    for o in outputs
                )
            elif hasattr(outputs, "shape"):
                return jax_to_torch(outputs, device)
            else:
                return outputs
        elif self.target_framework == "numpy":
            if isinstance(outputs, (list, tuple)):
                return type(outputs)(
                    numpy_to_torch(o, device) if isinstance(o, np.ndarray) else o
                    for o in outputs
                )
            elif isinstance(outputs, np.ndarray):
                return numpy_to_torch(outputs, device)
            else:
                return outputs
        else:
            return outputs


def ensure_device_compatibility(
    tensor: torch.Tensor, backend_device: str
) -> torch.Tensor:
    """
    Ensure tensor is on a device compatible with the backend.

    Args:
        tensor: Input tensor
        backend_device: Backend device type ('cpu', 'cuda', 'tpu')

    Returns:
        Tensor on compatible device
    """
    if backend_device == "cuda" and not tensor.is_cuda:
        return tensor.cuda()
    elif backend_device == "cpu" and tensor.is_cuda:
        return tensor.cpu()
    elif tensor.device.type == "mps":
        # MPS is not compatible with JAX, move to CPU
        return tensor.cpu()
    else:
        return tensor