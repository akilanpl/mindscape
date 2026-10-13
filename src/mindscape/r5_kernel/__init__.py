"""Unified research runtime. Routing is procedural; transfer remains NO_GO."""
from .contract import Request, Response
from .runtime import Kernel
__all__ = ['Request', 'Response', 'Kernel']
