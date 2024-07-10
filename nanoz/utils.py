#!/usr/bin/env python3

import subprocess
import time
import logging
from datetime import datetime
from functools import wraps

from torch import cuda


def version():
    """
    Read the VERSION file and return the version number of the code.

    Returns
    -------
    str
        Version number of the code.
    """
    with open("VERSION", "r") as f:
        return f.read().rstrip()


def is_notebook():
    """
    Check the code if it's running in the IPython notebook.
    
    Returns
    -------
    bool
        True if it's running in the IPython notebook. False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True   # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def timing(func):
    """
    Decorator for measuring execution duration of a function.

    Parameters
    ----------
    func : callable
        Function.
    """
    @wraps(func)
    def wrap(*args, **kw):
        start_time = datetime.now()
        result = func(*args, **kw)
        end_time = datetime.now()
        logging.info(f"Execution time of {func.__name__}: {end_time-start_time}\n")
        return result
    return wrap


def copy_doc(copy_func):
    """
    Decorator for copying the docstring of a function into another docstring of a function at the keyword '[COPYDOC]'.

    Parameters
    ----------
    copy_func : callable or property
        Function to copy.
    """
    def wrapper(func):
        func.__doc__ = func.__doc__.replace("[COPYDOC]", copy_func.__doc__)
        return func
    return wrapper


class KeyErrorMessage(Exception):
    """
    Override __repr__ to take into account line break in the message of KeyError exception...
    """
    def __repr__(self):
        return str(self)


def assign_free_gpus(max_gpus=1, wait=False, sleep_time=10, ban_process=None):
    """
    Assigns free gpus to the current process.

    Parameters
    ----------
    max_gpus : int, default=1
        The maximum number of gpus to assign.
    wait : bool, default=False
        Whether to wait until a GPU is free.
    sleep_time : int, default=10
        Sleep time (in seconds) to wait before checking GPUs, if wait=True.
    ban_process : str or list of str, default=None
        GPU is considered free if no process from ban_process is running on the gpu.

    Returns
    -------
    str
        GPUs id separated by ",".
    """

    def _check(gpu_count, ban_processes):
        """
        Check if GPUs are free.

        Parameters
        ----------
        gpu_count : int
            The number of GPUs available.
        ban_processes : str or list of str
            GPU is considered free if no process from ban_process is running on the gpu.

        Returns
        -------
        str
            GPUs id separated by ",".
        """
        gpu_ids = list()
        for gpu in range(0, gpu_count):
            # Get GPU information
            gpu_processes = cuda.list_gpu_processes(gpu)
            if 'GPU:' not in gpu_processes:
                raise RuntimeError(f'cuda.list_gpu_processes({gpu}): {gpu_processes}')
            gpu_processes = gpu_processes.split('\n')
            gpu_identification = [gpu_process for gpu_process in gpu_processes if 'GPU:' in gpu_process]
            gpu_id = gpu_identification[0].split('GPU:')[-1]

            # Free GPU (easy way)
            if 'no processes are running' in gpu_processes:
                gpu_ids.extend(gpu_id)
                continue

            # Check if the GPU is free according to the ban list of processes
            if ban_processes is not None:
                if isinstance(ban_processes, str):
                    ban_processes = [ban_processes]
                free_gpu = True
                gpu_processes = [gpu_process for gpu_process in gpu_processes if 'process' in gpu_process]
                for process in gpu_processes:
                    if free_gpu is False:
                        break
                    process_id = [int(s) for s in process.split() if s.isdigit()][0]
                    process_name = subprocess.check_output(f'ps -o cmd= {process_id}', shell=True)
                    process_name = process_name.decode('utf-8')
                    for exclude_process in ban_processes:
                        if exclude_process in process_name:
                            free_gpu = False
                            break
                if free_gpu:
                    gpu_ids.extend(gpu_id)

        free_gpus = gpu_ids[: min(max_gpus, len(gpu_ids))]
        free_gpus = ','.join(free_gpus)
        return free_gpus

    gpu_nb = cuda.device_count()
    logging.debug(f'{gpu_nb} GPU(s)')
    first = True
    while True:
        gpus_to_use = _check(gpu_nb, ban_process)
        if gpus_to_use or not wait:
            break
        if first:
            logging.info(f'No free GPUs found, retrying every {sleep_time}s')
            first = False
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError(f'No free GPUs found')
    logging.debug(f'Using GPU(s): {gpus_to_use}')
    return gpus_to_use
