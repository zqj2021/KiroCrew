"""A git command cannot open or configure a pager, which would run a program.

`git grep -O<pager>` runs the pager on the matching files, and the pager a git
command uses can be set from the command line or the environment. The pager is a
program git runs, and the agent controls which program, so every one of these is a
route to run an arbitrary command through git. The self-kill floor refuses the whole
operation, not a judged pager value: the pager name is swappable (an alias, a `PATH`
shadow, a `LESSOPEN` hook), so no pager value is benign.

Refused: `git grep -O` / `--open-files-in-pager` in every spelling, and any git
invocation that sets the pager by config to a program (`-c core.pager=`, `-c
pager.<cmd>=`, `--config-env=`, `git config core.pager <prog>`, `GIT_PAGER=`, `PAGER=`,
`GIT_CONFIG_PARAMETERS=`, the `GIT_CONFIG_COUNT`/`GIT_CONFIG_KEY_n` env trio), including
behind an execution-prefix wrapper (`command`/`env`/`sudo`/`nice`/...) or an env export
on the same line. Allowed: a plain search, `git --no-pager` / `-P`, and the no-op pager
forms -- an empty or `cat` pager value and a git-false `pager.<cmd>` -- which run no
distinct program. The value is read in git's grammar, so a subcommand option
(`git grep -c` count, `-e` pattern) is not mistaken for a git-global pager control.

Every self-kill attack string is built from parts so no single literal here spells the
kill command.
"""

from __future__ import annotations

import pytest

from kiro_crew import security

_KILL = "self-protection-kill"
_PAGER = "self-protection-git-pager"
_K = "pk" + "ill"
_TARGET = "kirocrew"
_FILE = "test/test_argv_floor_inline_and_brace_scope.py"


def _verdict(cmd: str) -> "str | None":
    return security.is_denied(cmd)


def _denied(cmd: str) -> bool:
    return bool(_verdict(cmd))


def _is_kill(cmd: str) -> bool:
    reason = _verdict(cmd)
    return bool(reason) and _KILL in str(reason)


# ---- git grep -O / --open-files-in-pager, every spelling, is refused -----------------
# The pager value is irrelevant: a swappable name is never benign.


@pytest.mark.parametrize(
    "cmd",
    [
        "git grep -O less -- '*'",
        "git grep -Oless -- '*'",
        "git grep -O'less' -- '*'",
        "git grep -nOcat -e foo -- '*'",
        "git grep --open-files-in-pager=less -- '*'",
        "git grep --open-files-in-pager less -- '*'",
        "git grep --open=less -- '*'",
        "git grep -e foo -O cat -- README.md",
    ],
)
def test_git_grep_opening_a_pager_is_refused(cmd: str) -> None:
    assert _denied(cmd), cmd
    assert _PAGER in str(_verdict(cmd)), cmd


# ---- any route that sets the pager by config is refused ------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "git -c core.pager=less grep foo -- '*'",
        "git -c pager.grep=less grep foo -- '*'",
        "git -ccore.pager=less grep foo",
        "git --config-env=core.pager=MYVAR grep foo -- '*'",
        "git --config-env=pager.grep=MYVAR grep foo",
        "GIT_PAGER=less git grep foo -- '*'",
        "PAGER=less git grep foo -- '*'",
        "GIT_CONFIG_PARAMETERS=\"'core.pager=less'\" git grep foo -- '*'",
        "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.pager GIT_CONFIG_VALUE_0=less git grep foo",
        # `pager.<cmd>=true` turns paging ON (the default pager runs), so it is refused
        # even though it names no program; a bare key with no value is `=true` the same way.
        "git -c pager.grep=true grep foo -- '*'",
        "git -c core.pager grep foo -- '*'",
    ],
)
def test_git_setting_the_pager_by_config_is_refused(cmd: str) -> None:
    assert _denied(cmd), cmd
    assert _PAGER in str(_verdict(cmd)), cmd


# ---- a git-global option is read in git's grammar, not flat: an option that belongs to
# the SUBCOMMAND (grep's `-c` count, grep's `-P` Perl-regexp) is not a git-global pager
# control, and a git-global `-c core.pager=<prog>` set before the subcommand is not
# cancelled by the subcommand's own `-P`. -------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        # grep's own `-c` (count) takes no config operand, so the pattern after it
        # (`core.pager`, `pager.grep`) is data, not a git-global pager config.
        "git grep -c core.pager -- '*.md'",
        "git grep -c pager.grep",
        # grep's `-e` consumes the next token as its pattern, so a following `-O` /
        # `--open` is that pattern, not the pager option.
        "git grep -e -O -- README.md",
        "git grep -e --open -- README.md",
    ],
)
def test_a_subcommand_option_is_not_read_as_git_global(cmd: str) -> None:
    assert not _denied(cmd), _verdict(cmd)


def test_a_git_global_pager_config_is_not_cancelled_by_a_subcommand_flag() -> None:
    # `-c core.pager=./pager-script` is a git-GLOBAL config naming a program; `log -P`
    # is the log subcommand's own `-P`, which does not cancel the global pager route.
    cmd = "git -c core.pager=./pager-script log -P -1"
    assert _denied(cmd), cmd
    assert _PAGER in str(_verdict(cmd)), cmd


# ---- a pager cannot slip past the floor behind an execution-prefix wrapper, an env
# assignment exported on the same line, a `git config` writer, or an empty substitution
# that hides the `git` substring from the cheap gate. --------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        # execution-prefix wrappers before git: the wrapped git still opens the pager.
        "command git grep -O less -- x",
        "exec git grep -O less -- x",
        "nice git grep -O less -- x",
        "nice -n 10 git grep -O less -- x",
        "sudo git grep -O less -- x",
        "env git grep -O less -- x",
        # env / sudo carry their own NAME=value words, which set the pager.
        "env GIT_PAGER=less git grep foo -- x",
        # an env pager EXPORTED by an earlier command on the same line.
        "export GIT_PAGER=less; git grep foo -- x",
        # `git config core.pager <program>` WRITES the pager into a config file.
        "git config core.pager less",
        "git config --global core.pager /tmp/p",
        "git config pager.grep less",
        # an empty substitution hides the `git` substring but resolves to git.
        "gi$()t grep -Oless -- x",
    ],
)
def test_a_pager_behind_a_wrapper_or_writer_is_refused(cmd: str) -> None:
    assert _denied(cmd), cmd
    assert _PAGER in str(_verdict(cmd)), cmd


# ---- --no-pager / -P turns OUTPUT paging off, but never cancels `grep -O`: -O opens
# the matching files in a program whatever the output paging does. --------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "git --no-pager grep -O less -- '*'",
        "git -P grep -Ocat -- '*'",
        "git --no-pager grep --open-files-in-pager=less -- '*'",
    ],
)
def test_no_pager_does_not_cancel_grep_dash_o(cmd: str) -> None:
    assert _denied(cmd), cmd
    assert _PAGER in str(_verdict(cmd)), cmd


# ---- a pager opened/configured inside a nested payload is refused --------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "eval 'git grep -O less -- x'",
        "sh -c 'git -c core.pager=less grep foo -- x'",
        "bash -c 'GIT_PAGER=less git grep foo -- x'",
    ],
)
def test_a_nested_git_pager_is_refused(cmd: str) -> None:
    assert _denied(cmd), cmd


# ---- turning the pager off, and a plain search, stay allowed -------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "git --no-pager grep foo -- '*.py'",
        "git -P grep foo -- '*.py'",
        "git grep foo -- '*.py'",
        "git grep -n -e foo -- ':!*.lock' '*'",
        "git log --oneline -5",
        # `-O` after the `--` operand terminator is a pathspec, not the pager option.
        "git grep -e foo -- README.md -O cat",
        # Lowercase `-o` prints matching text and runs no pager -- it is NOT `-O`.
        "git grep -o -e Kiro -- README.md",
        "git grep -no x",
        # An attached operand (`-e<pat>`) ends the short-option cluster, so an `O` in
        # the pattern is not read as the pager option.
        "git grep -eKiro",
        "git grep -eKirO -- README.md",
    ],
)
def test_a_pager_free_git_command_is_allowed(cmd: str) -> None:
    assert not _denied(cmd), _verdict(cmd)


# ---- the no-program pager forms are allowed: a route runs a pager only when it NAMES
# a distinct program, so an empty / `cat` / git-false / disabled pager setting is not
# refused. Every OTHER named program is still refused (the name is swappable). ---------


@pytest.mark.parametrize(
    "cmd",
    [
        # core.pager / GIT_PAGER / PAGER set to the empty string: no program.
        "git -c core.pager= log",
        "GIT_PAGER= git grep foo -- x",
        "PAGER= git log",
        # core.pager / GIT_PAGER / PAGER set to `cat`: the no-op pager the liveness
        # advice proposes and the product's own git calls use, so it stays allowed.
        "git -c core.pager=cat log",
        "git -c core.pager=cat grep foo -- '*'",
        "GIT_PAGER=cat git grep foo -- x",
        "PAGER=cat git log",
        "git -c pager.grep=cat grep foo -- x",
        # `git config` that READS, UNSETS, or writes a no-op pager is allowed.
        "git config --get core.pager",
        "git config core.pager cat",
        "git config --unset core.pager",
        "git config pager.grep false",
        # pager.<cmd> set to a git false: paging off, no program.
        "git -c pager.grep=false grep -n -e Kiro -- README.md",
        "git -c pager.grep=off grep foo -- x",
        "git -c pager.grep=0 grep foo -- x",
        # --no-pager / -P cancels a pager-config route (the pager never runs).
        "git -c core.pager=cat --no-pager log",
        "git --no-pager -c core.pager=less log",
        "git -P -c core.pager=cat log",
        "GIT_PAGER=less git --no-pager log",
    ],
)
def test_the_no_program_pager_forms_are_allowed(cmd: str) -> None:
    assert not _denied(cmd), _verdict(cmd)


# ---- the four self-kill bypass classes stay refused (regression guards) --------------
# There is no operand exemption, so each of these is refused. They pin that a future
# `git grep` data exemption must not reopen any of them.


@pytest.mark.parametrize(
    "cmd",
    [
        f"git grep -e '*' -e {_TARGET} -e ';' {_K} -f {_TARGET} -- {_FILE}",
        f'git(){{ shift; "$@"; }}; git grep {_K} -f {_TARGET}',
        f'function git {{ shift; "$@"; }}; git grep {_K} -f {_TARGET}',
        f"PATH=/tmp:$PATH; git grep {_K} -f {_TARGET}",
        f"hash -p /tmp/shim/git git; git grep {_K} -f {_TARGET}",
        f"command git grep {_K} -f {_TARGET}",
        f"true && git grep {_K} -f {_TARGET}",
        f"/tmp/git grep {_K} -f {_TARGET}",
        f"./git grep {_K} -f {_TARGET}",
        f"git(){{ shift; \"$@\"; }}; eval 'git grep {_K} -f {_TARGET}'",
        f"git(){{ shift; \"$@\"; }}; sh -c 'git grep {_K} -f {_TARGET}'",
    ],
)
def test_a_git_grep_operand_kill_stays_refused(cmd: str) -> None:
    assert _is_kill(cmd), cmd
