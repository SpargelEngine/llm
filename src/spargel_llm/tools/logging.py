from enum import Enum


class ANSIColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class LogType(Enum):
    UNKNOWN = 0
    INFO = 1
    SUCCESS = 2
    WARNING = 3
    ERROR = 4
    DEBUG = 5


LOG_COLOR_MAP = {
    LogType.UNKNOWN: ANSIColors.RESET,
    LogType.SUCCESS: ANSIColors.GREEN,
    LogType.INFO: ANSIColors.CYAN,
    LogType.WARNING: ANSIColors.BOLD + ANSIColors.BRIGHT_YELLOW,
    LogType.ERROR: ANSIColors.BOLD + ANSIColors.BRIGHT_RED,
    LogType.DEBUG: ANSIColors.MAGENTA,
}


def log(message, type: LogType = LogType.UNKNOWN):
    if type is LogType.UNKNOWN:
        print(message)
    else:
        print(f"{LOG_COLOR_MAP[type]}{message}{ANSIColors.RESET}")


def log_success(message):
    log(message, type=LogType.SUCCESS)


def log_info(message):
    log(message, type=LogType.INFO)


def log_warning(message):
    log(message, type=LogType.WARNING)


def log_error(message):
    log(message, type=LogType.ERROR)


def log_debug(message):
    log(message, type=LogType.DEBUG)
