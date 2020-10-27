import logging
import os
import sys
from contextlib import contextmanager

logger_is_setup = False
console_handler = None
file_handler = None


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


# colored stream handler for python logging framework (use the ColorStreamHandler class).
# from https://gist.github.com/mooware/a1ed40987b6cc9ab9c65
class ColorStreamHandler(logging.StreamHandler):
    DEFAULT = '\x1b[0m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    YELLOW = '\x1b[33m'
    BLUE = '\x1b[34m'
    MAGENDA = '\x1b[35m'
    CYAN = '\x1b[36m'

    CRITICAL = RED
    ERROR = RED
    WARNING = YELLOW
    INFO = GREEN
    DEBUG = CYAN
    DEBUGLLV = MAGENDA  # 5

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:
            return cls.CRITICAL
        elif level >= logging.ERROR:
            return cls.ERROR
        elif level >= logging.WARNING:
            return cls.WARNING
        elif level >= logging.INFO:
            return cls.INFO
        elif level >= logging.DEBUG:
            return cls.DEBUG
        elif level >= 5:
            return cls.DEBUGLLV
        else:
            return cls.DEFAULT

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def format(self, record):
        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return color + text + self.DEFAULT


def logger_setup(log_file, loggers, level):
    global logger_is_setup, console_handler, file_handler
    if logger_is_setup:
        print('Logger Already Setup, Check your code')
        return console_handler, file_handler, level

    file_handler = logging.FileHandler(log_file, 'w', 'utf-8')
    console_handler = ColorStreamHandler()

    formatter = logging.Formatter('[%(asctime)s] %(name)-9s:%(levelname)-8s: %(message)s',
                                  datefmt="%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    # console_handler.setLevel(level)

    file_handler.setFormatter(formatter)
    # file_handler.setLevel(level)

    for name in loggers:
        if isinstance(name, str):
            logging.getLogger(name).addHandler(console_handler)
            logging.getLogger(name).addHandler(file_handler)
            logging.getLogger(name).setLevel(level)
        elif isinstance(name, logging.Logger):
            name.addHandler(console_handler)
            name.addHandler(file_handler)
            name.setLevel(level)
    logger_is_setup = True
    return console_handler, file_handler, level


def logger_setup_extend(names, console_handler, file_handler, level):
    for name in names:
        if isinstance(name, str):
            logging.getLogger(name).addHandler(console_handler)
            logging.getLogger(name).addHandler(file_handler)
            logging.getLogger(name).setLevel(level)
        elif isinstance(name, logging.Logger):
            name.addHandler(console_handler)
            name.addHandler(file_handler)
            name.setLevel(level)
