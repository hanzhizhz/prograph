"""
Performance timing utilities for ProGraph

Provides lightweight timing instrumentation with hierarchical support
for identifying performance bottlenecks in the QA pipeline.
"""

import time
from typing import Optional


class TimingContext:
    """Context manager for timing code blocks with hierarchical support

    Example:
        with TimingContext("My operation"):
            # code to time
            pass
    """

    def __init__(self, name: str, parent: Optional['TimingContext'] = None, enabled: bool = True):
        """Initialize timing context

        Args:
            name: Name of the operation being timed
            parent: Parent timing context for hierarchical timing
            enabled: Whether timing is enabled
        """
        self.name = name
        self.parent = parent
        self.enabled = enabled
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.children = []

        if parent:
            parent.children.append(self)

    def __enter__(self):
        """Start timing when entering context"""
        if self.enabled:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log when exiting context"""
        if self.enabled:
            self.end_time = time.perf_counter()
            self.duration = self.end_time - self.start_time
            self._log_timing()
        return False  # Don't suppress exceptions

    def _log_timing(self):
        """Print structured timing log with hierarchical indentation"""
        indent = "  " * self._get_depth()
        print(f"{indent}⏱️  {self.name}: {self.duration:.3f}s")

    def _get_depth(self) -> int:
        """Calculate depth in timing hierarchy"""
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth


class TimingLogger:
    """Global timing logger with enable/disable support

    Example:
        logger = TimingLogger(enabled=True)
        with logger.time("Operation 1"):
            # code to time
            pass
    """

    def __init__(self, enabled: bool = True):
        """Initialize timing logger

        Args:
            enabled: Whether timing is enabled globally
        """
        self.enabled = enabled
        self.root_contexts = []

    def time(self, name: str, parent: Optional[TimingContext] = None) -> TimingContext:
        """Create a timing context

        Args:
            name: Name of the operation being timed
            parent: Parent timing context for hierarchical timing

        Returns:
            TimingContext that can be used as a context manager
        """
        context = TimingContext(name, parent, self.enabled)
        if not parent:
            self.root_contexts.append(context)
        return context

    def enable(self):
        """Enable timing globally"""
        self.enabled = True

    def disable(self):
        """Disable timing globally"""
        self.enabled = False
