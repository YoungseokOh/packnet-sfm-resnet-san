"""
Mock Horovod utilities for single GPU training
"""

import torch

# Single GPU mode - no Horovod
HAS_HOROVOD = False


def hvd_init():
    """Mock Horovod initialization"""
    print("🔧 Mock Horovod init (single GPU mode)")
    pass


def on_rank_0(func):
    """Decorator to run function only on rank 0"""
    def wrapper(*args, **kwargs):
        if rank() == 0:
            return func(*args, **kwargs)
    return wrapper


def rank():
    """Mock rank function - always return 0"""
    return 0


def size():
    """Mock size function - always return 1"""
    return 1


def world_size():
    """Mock world size - always return 1"""
    return 1


@on_rank_0
def print0(string='\n'):
    """Print only on rank 0"""
    print(string)


def reduce_value(value, average=True, name=""):
    """
    Mock reduce function for single GPU
    Just returns the input value unchanged
    
    Parameters
    ----------
    value : torch.Tensor
        Value to be "reduced"
    average : bool
        Whether values will be averaged (ignored in single GPU)
    name : str
        Value name (ignored)

    Returns
    -------
    value : torch.Tensor
        Same input value (no reduction needed)
    """
    return value


def allreduce(tensor, average=True, name=""):
    """Mock allreduce - just return the tensor"""
    return tensor


def broadcast_parameters(params, root_rank=0):
    """Mock broadcast parameters - no-op in single GPU"""
    pass


def broadcast_optimizer_state(optimizer, root_rank=0):
    """Mock broadcast optimizer state - no-op in single GPU"""
    pass


def DistributedOptimizer(optimizer, **kwargs):
    """Mock distributed optimizer - just return the original optimizer"""
    return optimizer


class Compression:
    """Mock compression class"""
    none = None


# Mock hvd object for backward compatibility
class MockHVD:
    @staticmethod
    def init():
        hvd_init()
    
    @staticmethod
    def rank():
        return 0
    
    @staticmethod
    def size():
        return 1
    
    @staticmethod
    def local_rank():
        return 0
    
    @staticmethod
    def allreduce(tensor, average=True, name=""):
        return tensor
    
    @staticmethod
    def broadcast_parameters(params, root_rank=0):
        pass
    
    @staticmethod
    def broadcast_optimizer_state(optimizer, root_rank=0):
        pass
    
    @staticmethod
    def DistributedOptimizer(optimizer, **kwargs):
        return optimizer
    
    class Compression:
        none = None


# Create mock hvd instance
hvd = MockHVD()
