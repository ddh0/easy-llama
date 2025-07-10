# test_utils.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

import sys
import datetime

from typing import Optional

class ANSI:
    """ANSI codes for terminal emulators"""

    BELL = '\a'

    CLEAR = '\x1bc\x1b[3J' # technically this is two ANSI codes in one

    # Standard colors
    FG_BLACK   = '\x1b[30m'
    BG_BLACK   = '\x1b[40m'
    FG_RED     = '\x1b[31m'
    BG_RED     = '\x1b[41m'
    FG_GREEN   = '\x1b[32m'
    BG_GREEN   = '\x1b[42m'
    FG_YELLOW  = '\x1b[33m'
    BG_YELLOW  = '\x1b[43m'
    FG_BLUE    = '\x1b[34m'
    BG_BLUE    = '\x1b[44m'
    FG_MAGENTA = '\x1b[35m'
    BG_MAGENTA = '\x1b[45m'
    FG_CYAN    = '\x1b[36m'
    BG_CYAN    = '\x1b[46m'
    FG_WHITE   = '\x1b[37m'
    BG_WHITE   = '\x1b[47m'

    # Bright colors
    FG_BRIGHT_BLACK   = '\x1b[90m'
    BG_BRIGHT_BLACK   = '\x1b[100m'
    FG_BRIGHT_RED     = '\x1b[91m'
    BG_BRIGHT_RED     = '\x1b[101m'
    FG_BRIGHT_GREEN   = '\x1b[92m'
    BG_BRIGHT_GREEN   = '\x1b[102m'
    FG_BRIGHT_YELLOW  = '\x1b[93m'
    BG_BRIGHT_YELLOW  = '\x1b[103m'
    FG_BRIGHT_BLUE    = '\x1b[94m'
    BG_BRIGHT_BLUE    = '\x1b[104m'
    FG_BRIGHT_MAGENTA = '\x1b[95m'
    BG_BRIGHT_MAGENTA = '\x1b[105m'
    FG_BRIGHT_CYAN    = '\x1b[96m'
    BG_BRIGHT_CYAN    = '\x1b[106m'
    FG_BRIGHT_WHITE   = '\x1b[97m'
    BG_BRIGHT_WHITE   = '\x1b[107m'

    # Modes
    MODE_RESET_ALL           = '\x1b[0m'
    MODE_BOLD_SET            = '\x1b[1m'
    MODE_BOLD_RESET          = '\x1b[22m'
    MODE_DIM_SET             = '\x1b[2m'
    MODE_DIM_RESET           = '\x1b[22m'
    MODE_ITALIC_SET          = '\x1b[3m'
    MODE_ITALIC_RESET        = '\x1b[23m'
    MODE_UNDERLINE_SET       = '\x1b[4m'
    MODE_UNDERLINE_RESET     = '\x1b[24m'
    MODE_BLINKING_SET        = '\x1b[5m'
    MODE_BLINKING_RESET      = '\x1b[25m'
    MODE_REVERSE_SET         = '\x1b[7m'
    MODE_REVERSE_RESET       = '\x1b[27m'
    MODE_HIDDEN_SET          = '\x1b[8m'
    MODE_HIDDEN_RESET        = '\x1b[28m'
    MODE_STRIKETHROUGH_SET   = '\x1b[9m'
    MODE_STRIKETHROUGH_RESET = '\x1b[29m'

def log(text: str, good: Optional[bool] = None) -> None:
    """Print the given text, prefixed with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %a %H:%M:%S.%f")[:-3]
    fmt_prefix_txt = ANSI.MODE_RESET_ALL + ANSI.MODE_BOLD_SET
    if good is not None:
        if good:
            fmt_prefix_txt += ANSI.FG_BRIGHT_GREEN
        else:
            fmt_prefix_txt += ANSI.FG_BRIGHT_RED
    print(
        f"{ANSI.MODE_RESET_ALL}{ANSI.MODE_BOLD_SET}{ANSI.FG_BRIGHT_BLACK}[{timestamp}]"
        f"{fmt_prefix_txt}TEST{ANSI.MODE_RESET_ALL}{ANSI.MODE_BOLD_SET}{ANSI.FG_BRIGHT_BLACK}:"
        f"{ANSI.MODE_RESET_ALL} {text}",
        end='\n',
        file=sys.stdout if good is not False else sys.stderr,
        flush=True
    )
