"""Bind a signed Python stdio launch to the gateway's selected bytecode cache."""

from __future__ import annotations

import os
import re
import sys
from typing import Any

from kiro_crew.platform_compat_runtime import _PYTHON_OPTIONS_WITH_VALUES

_PYTHON_COMMAND = re.compile(r"(?:python(?:\d+(?:\.\d+)*)?w?|py)(?:\.exe)?", re.IGNORECASE)


def with_python_cache_prefix(cfg: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Put the cache option last before Python's target, retaining authored flags.

    Python's ``-E`` and ``-I`` ignore the environment transport. Cache selection
    consumes an interpreter cache option during startup, so authored cache options
    are removed before the gateway emits its own. Other flags and arguments
    handed to the script/module are preserved. Non-Python commands keep their env
    transport and argv unchanged. Blocking work is owned by the cache allocator.
    """
    command = cfg.get("command")
    if not isinstance(command, str) or (
        command != sys.executable and not _PYTHON_COMMAND.fullmatch(os.path.basename(command))
    ):
        return cfg
    args = list(cfg.get("args") or [])
    options: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if not isinstance(arg, str):
            return cfg
        if arg in ("-", "--") or not arg.startswith("-"):
            break
        if arg.startswith("--"):
            consumed = 2 if arg in _PYTHON_OPTIONS_WITH_VALUES else 1
            options.extend(args[index : index + consumed])
            index += consumed
            continue
        # Short-option clusters stop at a launch selector or a value-bearing
        # option: the rest of -c/-m is the target, and of -W/-X is its value.
        target = False
        consumes_value = False
        cache_option = False
        keep_flag = arg
        for position, option in enumerate(arg[1:], start=1):
            if option in ("c", "m"):
                target = True
                break
            if option in ("W", "X"):
                consumes_value = position == len(arg) - 1
                value = (
                    args[index + 1]
                    if consumes_value and index + 1 < len(args)
                    else arg[position + 1 :]
                )
                cache_option = (
                    option == "X"
                    and isinstance(value, str)
                    and value.partition("=")[0] == "pycache_prefix"
                )
                if cache_option:
                    keep_flag = arg[:position]
                break
        if target:
            break
        consumed = 2 if consumes_value else 1
        if cache_option:
            if keep_flag != "-":
                options.append(keep_flag)
        else:
            options.extend(args[index : index + consumed])
        index += consumed
    return {
        **cfg,
        "args": [*options, "-X", f"pycache_prefix={prefix}", *args[index:]],
    }
