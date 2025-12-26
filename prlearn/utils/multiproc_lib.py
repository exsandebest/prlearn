try:
    import multiprocess as mp
except ImportError:
    import multiprocessing as mp

__all__ = ["mp"]
