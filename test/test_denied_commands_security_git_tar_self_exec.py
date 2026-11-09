"""ext / executed-config / linearity / alias tests for the git/tar self floor.

Split out of ``test_denied_commands_security_git_tar_self.py`` to keep each file
under the 2000-line cap. Same rule, own-host pin and helpers as that sibling;
this file holds the ``ext::`` decode, executed-value config keys, GIT_SSH env,
url-rewrite base, the bounded-work (linearity) tests and the alias/residual
cases.
"""

from __future__ import annotations

import pytest
from denied_commands_security_helpers import _own_address_probe_stays_local  # noqa: F401
from denied_commands_security_helpers import _denied_by, _rule_pattern  # noqa: F401

from kiro_crew import security
from kiro_crew.security import BUILTIN_DENIED_RULES  # noqa: F401
from kiro_crew.security import argv_floor as _argv_floor
from kiro_crew.security import compute_effective_denied, is_denied, is_safe_user_regex  # noqa: F401

pytestmark = pytest.mark.usefixtures("fresh_darwin_libc_handles")


class TestGitTarSelfExec:
    """ext / executed-config / linearity / alias cases for the git/tar floor."""

    _RULE = "sandbox-escape-git-tar-self"
    #: A command a git or tar transport runs is re-entered through every argv
    #: floor, and the refusal names the row that refused it: a carried ``ssh``
    #: that dials this host is reported under the ssh floor's own row.
    _SSH_RULE = "sandbox-escape-ssh-self"
    _DIAL_RULES = frozenset({_RULE, _SSH_RULE})

    @staticmethod
    def _effective():
        return list(compute_effective_denied(BUILTIN_DENIED_RULES, (), False, (), ()))

    @pytest.fixture(autouse=True)
    def _pin_own_host_cache(self, request, _floor_monkeypatch):
        # Same pin as the ssh-self suite: a fixed own-host cache with the
        # resolved-once latch set, so no command here spawns the real own-host
        # resolver thread or blocks on DNS, and the DNS-alias verdict layer is
        # stubbed to "not self" so remote-host allow cases stay allowed.
        own_mp = pytest.MonkeyPatch()
        request.addfinalizer(own_mp.undo)
        own = security.socket.gethostname().strip().lower()
        pinned = frozenset(name for name in {own, own.split(".", 1)[0]} if name)
        _floor_monkeypatch.setattr(_argv_floor, "_OWN_HOST_NAMES_CACHE", pinned)
        _floor_monkeypatch.setattr(_argv_floor, "_OWN_HOST_RESOLVE_DONE", True)
        _floor_monkeypatch.setattr(_argv_floor, "_NETLINK_ADDRS_PUBLISHED", True)
        own_mp.setattr(_argv_floor, "_OWN_HOST_RESOLVE_NEXT_TRY", float("inf"))
        _floor_monkeypatch.setattr(_argv_floor, "_resolved_host_verdict", lambda host, **_kw: False)

    @pytest.mark.parametrize(
        ("cmd", "rule"),
        [
            (
                "git -c protocol.ext.allow=always ls-remote " "'ext::git ls-remote localhost:x'",
                _RULE,
            ),
            (
                "git -c protocol.ext.allow=always ls-remote " "'ext::ssh localhost id'",
                _SSH_RULE,
            ),
            (
                "git -c protocol.ext.allow=always ls-remote "
                "'ext::git ls-remote far.example.com:x'",
                None,
            ),
            (
                "git -c protocol.ext.allow=always ls-remote 'ext::sh -c id'",
                None,
            ),
        ],
        ids=["git-self", "ssh-self-control", "git-remote-control", "shell-control"],
    )
    def test_git_ext_commands_run_through_both_command_floors(self, cmd, rule):
        assert _denied_by(cmd) == rule, cmd

    def test_git_ext_escaped_spaces_to_self_are_denied(self):
        cmd = (
            "git -c protocol.ext.allow=always ls-remote "
            "'ext::ssh -oProxyCommand=ssh% localhost% id far.example.com repo'"
        )
        assert _denied_by(cmd) == self._SSH_RULE

    def test_git_ext_escaped_spaces_to_remote_are_allowed(self):
        cmd = (
            "git -c protocol.ext.allow=always ls-remote "
            "'ext::ssh -oProxyCommand=ssh% bastion.example.com far.example.com repo'"
        )
        assert _denied_by(cmd) not in self._DIAL_RULES

    def test_git_ext_literal_percent_is_decoded(self):
        command = _argv_floor._git_ext_command(
            "ext::ssh -oIdentityFile=key%%name far.example.com", []
        )
        assert command == "ssh -oIdentityFile=key%name far.example.com"

    # An unquoted ``ext::`` payload runs on over the operands after it, options
    # included, up to the next ``ext::`` operand.
    @pytest.mark.parametrize(
        ("cmd", "rule"),
        [
            ("git clone ext::x ext::ssh localhost id", _SSH_RULE),
            ("git clone ext::ssh localhost ext::x", _SSH_RULE),
            ("git clone ext::ssh -p 22 localhost id", _SSH_RULE),
            ("git clone ext::ssh --upload-pack=ext::x localhost", _SSH_RULE),
            ("git clone ext::ssh far.example.com --upload-pack=ext::ssh% localhost", _SSH_RULE),
            ("git clone ext::ssh far.example.com ext::x localhost", None),
        ],
        ids=[
            "second-operand",
            "before-next-operand",
            "over-options",
            "over-option-value",
            "option-value-payload",
            "ends-at-next-operand",
        ],
    )
    def test_an_unquoted_ext_payload_runs_to_the_next_ext_operand(self, cmd, rule):
        assert _denied_by(cmd) == rule, cmd

    # ── executable git config values: the ``!`` and shell forms ──
    # git EXECUTES the value of each key below.  For a ``credential.helper`` /
    # ``credential.<url>.helper`` and an ``alias`` body git strips a leading
    # ``!`` and runs the remainder through the shell; for every other key it
    # runs the value as a shell command verbatim, where a leading ``! word``
    # still runs ``word``.  So a value that dials THIS host is refused in BOTH
    # the ``!ssh localhost id`` form and the bare ``ssh localhost id`` form,
    # reached either as ``-c key=value`` or through the numbered
    # ``GIT_CONFIG_COUNT`` / ``GIT_CONFIG_KEY_n`` / ``GIT_CONFIG_VALUE_n``
    # environment.  Source: ``git help config`` for the executing keys and
    # ``git help cli`` / ``git help environment`` for the numbered arrival form.
    _EXECUTED_CONFIG_KEYS = (
        "credential.helper",
        "credential.https://example.com.helper",
        "core.sshCommand",
        "core.gitProxy",
        "core.pager",
        "core.editor",
        "core.fsmonitor",
        "diff.d.command",
        "diff.d.textconv",
        "merge.m.driver",
        "filter.f.clean",
        "filter.f.smudge",
        "filter.f.process",
        "gpg.program",
        "gpg.x509.program",
        "sequence.editor",
        "pager.log",
        "alias.co",
    )

    @staticmethod
    def _dash_c_value(key: str, value: str) -> str:
        return f"git -c '{key}={value}' fetch https://example.com/x"

    @staticmethod
    def _numbered_env_value(key: str, value: str) -> str:
        return (
            f"GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0='{key}'"
            f" GIT_CONFIG_VALUE_0='{value}' git fetch https://example.com/x"
        )

    @pytest.mark.parametrize(
        "value", ["!ssh localhost id", "ssh localhost id"], ids=["bang", "shell"]
    )
    @pytest.mark.parametrize("arrival", ["dash_c", "numbered_env"])
    @pytest.mark.parametrize("key", _EXECUTED_CONFIG_KEYS)
    def test_executed_config_value_dialing_self_is_refused(self, key, arrival, value):
        build = self._dash_c_value if arrival == "dash_c" else self._numbered_env_value
        cmd = build(key, value)
        # The carried ``ssh localhost id`` is reported under the ssh floor's row.
        assert _denied_by(cmd) == self._SSH_RULE, cmd

    # The ``GIT_SSH`` / ``GIT_SSH_COMMAND`` / ``GIT_PROXY_COMMAND`` transport
    # commands are the same ``!``/shell pair, delivered only through the
    # environment.
    @pytest.mark.parametrize(
        "value", ["!ssh localhost id", "ssh localhost id"], ids=["bang", "shell"]
    )
    @pytest.mark.parametrize("name", ["GIT_SSH_COMMAND", "GIT_PROXY_COMMAND", "GIT_SSH"])
    def test_executed_ssh_env_command_dialing_self_is_refused(self, name, value):
        cmd = f"{name}='{value}' git fetch https://example.com/x"
        assert _denied_by(cmd) == self._SSH_RULE, cmd

    # A ``url.<base>.insteadOf`` / ``pushInsteadOf`` rewrite whose base is an
    # ``ext::`` command dials this host when that command does (the ssh/scp
    # bases are covered by ``test_url_rewrite_to_self_is_denied``); a base to a
    # remote host stays allowed.
    @pytest.mark.parametrize(
        ("cmd", "rule"),
        [
            ("git -c 'url.ext::ssh localhost id.insteadOf=ex:' fetch ex:repo", _SSH_RULE),
            (
                "git -c 'url.ext::ssh localhost id.pushInsteadOf=ex:' push ex:repo feat",
                _SSH_RULE,
            ),
            ("git -c 'url.ext::ssh far.example.com id.insteadOf=ex:' fetch ex:repo", None),
        ],
        ids=[
            "ext-insteadof-self",
            "ext-pushinsteadof-self",
            "ext-insteadof-remote",
        ],
    )
    def test_url_rewrite_ext_base_dialing_self_is_refused(self, cmd, rule):
        assert _denied_by(cmd) == rule, cmd

    # Controls: an executed value git runs that does NOT dial this host stays
    # allowed, including a bare ``!`` helper and a ``!``-prefixed command naming
    # a remote host (the ``!`` strip must not read a far host as self).
    @pytest.mark.parametrize(
        "cmd",
        [
            "git -c 'credential.helper=!echo hi' fetch https://example.com/x",
            "git -c credential.helper=store fetch https://example.com/x",
            "git -c core.pager=less fetch https://example.com/x",
            "git -c 'credential.helper=!ssh far.example.com id' fetch https://example.com/x",
            "git -c 'core.sshCommand=!ssh far.example.com id' fetch far.example.com:x",
            (
                "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper"
                " GIT_CONFIG_VALUE_0='!echo hi' git fetch https://example.com/x"
            ),
        ],
        ids=[
            "bang-echo",
            "builtin-store",
            "pager-less",
            "bang-ssh-remote",
            "sshcommand-bang-remote",
            "numbered-bang-echo",
        ],
    )
    def test_executed_config_controls_stay_allowed(self, cmd):
        assert _denied_by(cmd) is None, cmd

    @pytest.fixture
    def work_meter(self, monkeypatch):
        """Count the two work units the quadratic paths inflate.

        ``following`` sums ``len(following)`` over every ``_git_ext_command``
        payload read -- the ext-payload operands one ``ext::`` read consumes,
        which the ext bound keeps at the next ext operand.  ``reentries`` counts
        dispatcher re-entries (``deny_gate._reentered_floor_hit``), which the
        re-entry memo collapses for a command many invocations carry.  Both are
        exact, host-independent counts: no wall clock, no coverage sensitivity.
        """
        from kiro_crew.security import deny_gate as _deny_gate

        counts = {"following": 0, "reentries": 0}
        real_ext = _argv_floor._git_ext_command
        real_hit = _deny_gate._reentered_floor_hit

        def ext(operand, following):
            counts["following"] += len(following)
            return real_ext(operand, following)

        def hit(command):
            counts["reentries"] += 1
            return real_hit(command)

        monkeypatch.setattr(_argv_floor, "_git_ext_command", ext)
        monkeypatch.setattr(_deny_gate, "_reentered_floor_hit", hit)

        class _Meter:
            def reset(self):
                counts["following"] = 0
                counts["reentries"] = 0

            @property
            def following(self):
                return counts["following"]

            @property
            def reentries(self):
                return counts["reentries"]

        return _Meter()

    # Each of n pure ``ext::`` operands bounds its payload to the NEXT ext
    # operand, so no operand re-reads the tail: the ext-payload words read stay
    # at zero however many operands there are.  Restoring the per-operand
    # re-read of every later operand (the quadratic ``ext::`` path) makes each
    # operand decode the whole tail, so the payload words read go O(n^2) and the
    # command is refused once the carried-read budget trips.  Reading the work
    # count and the ``None`` verdict catches both, with no wall clock.
    def test_many_ext_operands_are_read_in_linear_work(self, work_meter):
        for n in (1000, 2000):
            cmd = "git clone " + " ".join(["ext::x"] * n)
            work_meter.reset()
            verdict = is_denied(cmd)
            assert verdict is None, (n, verdict)
            # The ext-payload words read never exceed the operand count; the
            # quadratic restore drives this past it (then to the budget cap).
            assert work_meter.following <= len(cmd.split()), (n, work_meter.following)

    #: Each worst case is read with its re-entry work bounded.  A repeated or
    #: carried command re-enters the dispatcher a bounded number of times (the
    #: re-entry memo collapses a command many invocations carry); the largest
    #: healthy case re-enters 35 times, well under this ceiling.  Restoring a
    #: quadratic path drives a case that must be ALLOW to the carried-read
    #: budget and so to a DENY verdict -- caught by the verdict check below --
    #: while this ceiling catches an unbounded re-entry runaway.  No wall clock.
    _REENTRY_WORK_CEILING = 128
    _GLUED = " -o" + "a" * 250

    # Each command stays under ``MAX_SCANNABLE_COMMAND_CHARS`` and every git or
    # tar argv in it under ``_GIT_TAR_WORD_CAP``.  The ``denied`` cases are
    # refused because the read passes ``_CARRIED_READ_CHAR_BUDGET`` (fail closed).
    @pytest.mark.parametrize(
        ("cmd", "denied"),
        [
            ("git" + " -c alias.x=status" * 1000 + " x", False),
            ("git" + "".join(f" -c alias.x=s{i}" for i in range(1000)) + " x", False),
            ("git" + " -c 'alias.x=!echo'" * 900 + " x", False),
            ("X=status git" + " --config-env=alias.x=X" * 800 + " x", False),
            ('GIT_CONFIG_PARAMETERS="' + " 'alias.x=status'" * 1000 + '" git x', False),
            (
                "".join(
                    f"GIT_CONFIG_KEY_{i}=alias.x GIT_CONFIG_VALUE_{i}=s{i} " for i in range(380)
                )
                + "git x",
                True,
            ),
            ("git x" + "".join(f" -c alias.x=s{i}" for i in range(1000)), False),
            ("git clone" + " ext::x" * 2800, False),
            ("git clone ext::x" + " --u=ext::x" * 1800, False),
            ("git" + " -c core.sshCommand=ssh" * 800 + " fetch origin", False),
            ("export GIT_SSH_COMMAND='ssh" + _GLUED * 30 + "'; " + "git fetch o; " * 800, False),
            ("tar -cf out.tar" + " --rsh-command=ssh" * 1000 + " dir", False),
            ("git clone" + " --branch=a" * 1800 + " https://github.com/x/y", False),
            ("git clone" + " --upload-pack=x" * 1200 + " /local/repo", False),
            (
                "".join(f"export TAR_OPTIONS=-v{i}; " for i in range(350))
                + "tar -cf out.tar"
                + " a" * 2400,
                False,
            ),
            ("export TAR_OPTIONS='" + " -v" * 1500 + "'; " + "tar -cf o d; " * 600, False),
            ("export TAR_OPTIONS='--label=" + "a" * 9000 + "'; " + "tar; " * 2200, False),
            (
                'export GIT_CONFIG_PARAMETERS="'
                + " 'a.b=c'" * 1100
                + '"; '
                + "git fetch o; " * 750,
                False,
            ),
            (
                'export GIT_CONFIG_PARAMETERS="' + " 'a.b=c'" * 1000 + '"; ' + "git x; " * 1500,
                False,
            ),
            (
                "".join(f"export X=s{i}; " for i in range(500))
                + "git"
                + "".join(f" --config-env=k{i}=X" for i in range(500))
                + " fetch origin",
                False,
            ),
            (
                "".join(
                    f"export GIT_CONFIG_KEY_0=k{i}; export GIT_CONFIG_VALUE_0=v{i}; "
                    for i in range(150)
                )
                + "git fetch o; " * 700,
                False,
            ),
            ("export A='" + _GLUED * 30 + "'; " + "git -c alias.x=status x; " * 450, False),
            (
                "export A='"
                + _GLUED * 30
                + "'; "
                + "".join(f"git -c alias.x={i} x; " for i in range(450)),
                True,
            ),
            (
                "export A='"
                + _GLUED * 30
                + "'; "
                + "".join(f"git -c 'alias.x=!echo {i}' x; " for i in range(400)),
                True,
            ),
        ],
        ids=[
            "repeated-alias",
            "redefined-alias",
            "repeated-shell-alias",
            "repeated-config-env-alias",
            "repeated-parameters-alias",
            "numbered-alias-definitions",
            "post-verb-alias-definitions",
            "ext-operands",
            "ext-option-values",
            "repeated-ssh-command-config",
            "exported-ssh-command-many-gits",
            "repeated-tar-rsh-command",
            "attached-branch",
            "attached-upload-pack",
            "tar-options-values",
            "exported-tar-options-many-tars",
            "exported-long-tar-options-many-tars",
            "exported-parameters-many-gits",
            "exported-parameters-many-alias-lookups",
            "config-env-name-values",
            "exported-numbered-product-many-gits",
            "exported-env-same-alias-many-gits",
            "exported-env-distinct-aliases",
            "exported-env-distinct-shell-aliases",
        ],
    )
    def test_worst_case_reads_finish_in_bounded_work(self, cmd, denied, work_meter):
        from kiro_crew.security.paths import MAX_SCANNABLE_COMMAND_CHARS

        assert len(cmd) <= MAX_SCANNABLE_COMMAND_CHARS, len(cmd)
        assert len(cmd.split()) <= _argv_floor._GIT_TAR_WORD_CAP, len(cmd.split())
        work_meter.reset()
        verdict = is_denied(cmd)
        assert (verdict is not None) is denied
        assert work_meter.reentries <= self._REENTRY_WORK_CEILING, work_meter.reentries

    # git reads the numbered environment config, then ``GIT_CONFIG_PARAMETERS``
    # (the inherited value, then this invocation's own ``-c`` / ``--config-env``
    # options), and runs the LAST ``alias.<verb>`` it read.  Each pair is one
    # shape twice: a later harmless definition is allowed, and the same two
    # definitions with the self-dialing one last are refused.
    _LAST_ALIAS_DEFINITION_PAIRS = [
        (
            "c-then-c",
            "git -c alias.x='ls-remote localhost:x' -c alias.x=status x",
            "git -c alias.x=status -c alias.x='ls-remote localhost:x' x",
        ),
        (
            "config-env-then-c",
            "X='ls-remote localhost:x' git --config-env=alias.x=X -c alias.x=status x",
            "X='ls-remote localhost:x' git -c alias.x=status --config-env=alias.x=X x",
        ),
        (
            "parameters-then-c",
            "GIT_CONFIG_PARAMETERS=\"'alias.x=ls-remote localhost:x'\" git -c alias.x=status x",
            "GIT_CONFIG_PARAMETERS=\"'alias.x=status'\" git -c alias.x='ls-remote localhost:x' x",
        ),
        (
            "parameter-then-parameter",
            "GIT_CONFIG_PARAMETERS=\"'alias.x=ls-remote localhost:x' 'alias.x=status'\" git x",
            "GIT_CONFIG_PARAMETERS=\"'alias.x=status' 'alias.x=ls-remote localhost:x'\" git x",
        ),
        (
            "numbered-then-c",
            "GIT_CONFIG_KEY_0=alias.x GIT_CONFIG_VALUE_0='ls-remote localhost:x' "
            "git -c alias.x=status x",
            "GIT_CONFIG_KEY_0=alias.x GIT_CONFIG_VALUE_0=status "
            "git -c alias.x='ls-remote localhost:x' x",
        ),
        (
            "shell-then-c",
            "git -c 'alias.x=!ssh localhost' -c alias.x=status x",
            "git -c alias.x=status -c 'alias.x=!ssh localhost' x",
        ),
        (
            "loop-then-c",
            "git -c alias.a=a -c alias.a=status a",
            "git -c alias.a=status -c alias.a=a a",
        ),
    ]

    @pytest.mark.parametrize(
        ("cmd", "denied"),
        [
            *(
                pytest.param(cmd, denied, id=f"{name}-{'reversed' if denied else 'allowed'}")
                for name, allowed, reversed_ in _LAST_ALIAS_DEFINITION_PAIRS
                for cmd, denied in ((allowed, False), (reversed_, True))
            ),
            pytest.param(
                "GIT_CONFIG_KEY_0=alias.x GIT_CONFIG_VALUE_0='ls-remote localhost:x' "
                "GIT_CONFIG_KEY_1=alias.x GIT_CONFIG_VALUE_1=status git x",
                True,
                id="every-numbered-definition-is-read",
            ),
            pytest.param(
                "git -c alias.x=status x -c alias.x='ls-remote localhost:x'",
                True,
                id="post-verb-definition-is-read",
            ),
        ],
    )
    def test_the_last_alias_definition_git_reads_decides(self, cmd, denied):
        assert security._is_git_tar_to_self(cmd) is denied, cmd
        if denied:
            assert _denied_by(cmd) in self._DIAL_RULES, cmd
        else:
            assert _denied_by(cmd) is None, cmd

    def test_a_read_past_the_carried_character_budget_is_refused(self, monkeypatch):
        cmd = "git -c alias.x=status x; git -c alias.y=status y"
        assert not security._is_git_tar_to_self(cmd)
        # The first rebuilt alias command fits, the second one does not.
        first = len("git -c alias.x=status status")
        monkeypatch.setattr(_argv_floor, "_CARRIED_READ_CHAR_BUDGET", first)
        assert security._is_git_tar_to_self(cmd)

    def test_a_git_argv_past_the_word_cap_is_refused_unread(self):
        cap = _argv_floor._GIT_TAR_WORD_CAP
        # ``origin`` and the ``a`` refspecs: exactly the cap, then one word past it.
        assert _denied_by("git fetch origin" + " a" * (cap - 1)) is None
        assert _denied_by("git fetch origin" + " a" * cap) == self._RULE

    def test_a_command_value_past_the_word_cap_is_refused_unread(self):
        cap = _argv_floor._GIT_TAR_WORD_CAP
        value = "ssh" + " a" * (cap - 1)
        assert _argv_floor._value_names_literal_self(value) is False
        assert _argv_floor._value_names_literal_self(value + " a") is None
        remote = " git fetch far.example.com:x"
        assert _denied_by(f"GIT_SSH_COMMAND='{value}'" + remote) is None
        assert _denied_by(f"GIT_SSH_COMMAND='{value} a'" + remote) == self._RULE

    @pytest.mark.parametrize(
        "cmd",
        [
            "GIT_PROXY_COMMAND='ssh localhost' git ls-remote git://far.example.com/repo",
            "git -c core.gitProxy='ssh localhost' ls-remote git://far.example.com/repo",
        ],
        ids=["environment", "config"],
    )
    def test_git_proxy_commands_to_self_are_denied(self, cmd):
        assert _denied_by(cmd) == self._SSH_RULE, cmd

    def test_invoked_shell_alias_to_self_is_denied(self):
        assert _denied_by("git -c alias.x='!ssh localhost' x") == self._SSH_RULE

    def test_uninvoked_shell_alias_to_self_is_allowed(self):
        cmd = "git -c alias.x='!ssh localhost' status"
        assert _denied_by(cmd) not in self._DIAL_RULES

    @pytest.mark.parametrize(
        ("cmd", "rule"),
        [
            ("git -c alias.x='ls-remote localhost:x' x", _RULE),
            ("X='ls-remote localhost:x' git --config-env=alias.x=X x", _RULE),
            (
                "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.x "
                "GIT_CONFIG_VALUE_0='ls-remote localhost:x' git x",
                _RULE,
            ),
            (
                "GIT_CONFIG_PARAMETERS=\"'alias.x=ls-remote localhost:x'\" git x",
                _RULE,
            ),
            ("git -c alias.x='!ssh localhost' x", _SSH_RULE),
            ("git -c alias.x='status' x", None),
            ("git -c alias.x='ls-remote far.example.com:x' x", None),
            ("git -c alias.x='ls-remote localhost:x' status", None),
            (
                "git -c alias.x='ssh localhost' ls-remote far.example.com:x",
                _SSH_RULE,
            ),
        ],
        ids=[
            "c-ordinary-self",
            "config-env-ordinary-self",
            "numbered-env-ordinary-self",
            "parameters-ordinary-self",
            "shell-self-control",
            "local-control",
            "remote-control",
            "ordinary-unused-control",
            "unused-alias-over-refusal-residual",
        ],
    )
    def test_invoked_ordinary_aliases_are_inspected(self, cmd, rule):
        # The final case pins the packaged key-agnostic command-value residual:
        # a transport invocation still inspects an unused alias value as a command.
        assert _denied_by(cmd) == rule, cmd

    @pytest.mark.parametrize(
        "cmd",
        [
            "GIT_CONFIG=/work/config git ls-remote far:x",
            "GIT_CONFIG_GLOBAL=/work/config git ls-remote far:x",
            "GIT_CONFIG_SYSTEM=/work/config git ls-remote far:x",
            "git -c include.path=/work/config ls-remote far:x",
            "GIT_DIR=/work/repo/.git git ls-remote far:x",
            "GIT_COMMON_DIR=/work/repo/.git git ls-remote far:x",
            "HOME=/work/home XDG_CONFIG_HOME=/work/config git ls-remote far:x",
        ],
        ids=[
            "config",
            "global",
            "system",
            "include-path",
            "git-dir",
            "common-dir",
            "home-and-xdg",
        ],
    )
    def test_file_backed_config_channels_remain_residual(self, cmd):
        assert _denied_by(cmd) not in self._DIAL_RULES, cmd

    @pytest.mark.parametrize(
        "cmd",
        [
            "git -c protocol.foo.allow=always ls-remote foo::localhost",
            "git -c remote.x.vcs=foo ls-remote x",
            "GIT_EXEC_PATH=/work/helpers git ls-remote foo::localhost",
        ],
        ids=["custom-scheme", "remote-vcs", "exec-path"],
    )
    def test_remote_helper_indirection_remains_residual(self, cmd):
        assert _denied_by(cmd) not in self._DIAL_RULES, cmd

    def test_transport_wrapper_indirection_remains_residual(self):
        cmd = "GIT_SSH=/work/self-routing-wrapper git ls-remote far:x"
        assert _denied_by(cmd) not in self._DIAL_RULES

    @pytest.mark.parametrize(
        "cmd",
        [
            "GIT_SSH_VARIANT=ssh git ls-remote far:x",
            "git -c ssh.variant=ssh ls-remote far:x",
        ],
        ids=["environment", "config"],
    )
    def test_ssh_variant_does_not_choose_a_target(self, cmd):
        assert _denied_by(cmd) not in self._DIAL_RULES, cmd

    @pytest.mark.parametrize(
        "cmd",
        [
            "GIT_ALLOW_PROTOCOL=ext git ls-remote ext::id",
            "GIT_PROTOCOL_FROM_USER=1 git ls-remote ext::id",
        ],
        ids=["allow-protocol", "from-user"],
    )
    def test_protocol_permission_does_not_hide_the_target(self, cmd):
        assert _denied_by(cmd) not in self._DIAL_RULES, cmd

    def test_opt_out_disables_both_tiers(self):
        effective = compute_effective_denied(BUILTIN_DENIED_RULES, (self._RULE,), False, (), ())
        assert is_denied("git clone localhost:x", denied_regexes=list(effective)) is None
        assert is_denied("tar -cf localhost:a.tar dir", denied_regexes=list(effective)) is None

    # Every command a git/tar invocation runs, and the git invocation an invoked
    # alias becomes, is re-entered through the whole argv-floor dispatcher.
    @pytest.mark.parametrize(
        ("cmd", "rule"),
        [
            # An ordinary alias keeps the invocation's own operands and config.
            ("git -c alias.c=clone c localhost:x", _RULE),
            ("git -c alias.f=fetch f localhost:x", _RULE),
            (
                "git -c alias.probe='ls-remote selftarget' "
                "-c remote.selftarget.url=localhost:/srv/repo probe",
                _RULE,
            ),
            ("GIT_SSH_COMMAND='ssh localhost' git -c alias.f=fetch f origin", _SSH_RULE),
            ("git -c alias.c='clone ssh://local%68ost/x' c", _RULE),
            ("git -c alias.x='ls-remote localhost:x' x", _RULE),
            ("git -c alias.x='status' x", None),
            ("GIT_SSH_COMMAND='ssh localhost' git -c alias.s=status s", None),
            ("git -c alias.c=clone c far.example.com:x", None),
            ("git -c alias.x='ls-remote localhost:x' status", None),
            # A ``!`` alias runs its shell command with the operands appended and
            # git's ``-c`` config in the child's environment.
            ("git -c 'alias.c=!git clone localhost:x' c", _RULE),
            ("git -c 'alias.c=!git clone' c localhost:x", _RULE),
            ("git -c core.sshCommand='ssh localhost' -c 'alias.x=!git fetch origin' x", _SSH_RULE),
            ("git -c core.sshCommand='ssh localhost' -c 'alias.x=!git status' x", None),
            ("git -c 'alias.c=!git clone far.example.com:x' c", None),
            # Nested aliases resolve, through either kind.
            ("git -c alias.a=b -c alias.b=c -c alias.c='ls-remote localhost:x' a", _RULE),
            ("git -c 'alias.a=!git b' -c alias.b='ls-remote localhost:x' a", _RULE),
            ("git -c alias.a=b -c alias.b=c -c alias.c=status a", None),
        ],
        ids=[
            "verb-plus-operand-clone",
            "verb-plus-operand-fetch",
            "alias-plus-remote-url-config",
            "alias-plus-ssh-command-env",
            "alias-encoded-host",
            "operand-in-body",
            "local-control",
            "env-local-verb-control",
            "remote-operand-control",
            "unused-alias-residual-control",
            "shell-alias-git-clone",
            "shell-alias-operand",
            "shell-alias-inherits-config",
            "shell-alias-local-verb-control",
            "shell-alias-remote-control",
            "nested-ordinary",
            "nested-shell-to-ordinary",
            "nested-local-control",
        ],
    )
    def test_invoked_alias_is_rebuilt_and_reentered(self, cmd, rule):
        assert security._is_git_tar_to_self(cmd) is (rule is not None), cmd
        assert _denied_by(cmd) == rule, cmd

    @pytest.mark.parametrize(
        "cmd",
        [
            "git -c alias.a=a a",
            "git -c alias.a=b -c alias.b=a a",
            "git -c alias.a='a x' a",
        ],
        ids=["self", "two-cycle", "growing"],
    )
    def test_alias_cycle_is_refused(self, cmd):
        # git rejects an alias loop; the floor refuses it rather than unroll it.
        assert _denied_by(cmd) == self._RULE, cmd

    def test_alias_chain_past_the_reentry_limit_is_refused(self):
        def chain(length: int) -> str:
            links = [f"-c alias.a{n}=a{n + 1}" for n in range(length - 1)]
            return "git " + " ".join([*links, f"-c alias.a{length - 1}=status", "a0"])

        within = chain(_argv_floor._FUNCTION_BIND_CAP)
        assert not security._is_git_tar_to_self(within), within
        beyond = chain(_argv_floor._FUNCTION_BIND_CAP + 1)
        assert security._is_git_tar_to_self(beyond), beyond

    @pytest.mark.parametrize(
        ("cmd", "rule"),
        [
            ("git ls-remote --upload-pack='git ls-remote localhost:/srv/repo' .", _RULE),
            ("git clone --upload-pack='ssh localhost id' /local/repo", _SSH_RULE),
            ("git ls-remote --receive-pack='git ls-remote localhost:x' .", _RULE),
            ("git fetch -u 'git ls-remote localhost:x' origin", _RULE),
            ("git archive --remote=far:x --exec='git ls-remote localhost:x' HEAD", _RULE),
            ("git ls-remote --upload-pack='git clone git+ssh://localhost/x' .", _RULE),
            ("git ls-remote --upload-pack='git clone [localhost:22]:x' .", _RULE),
            (
                "git ls-remote --upload-pack='git archive --remote=ssh://local%68ost/x HEAD' .",
                _RULE,
            ),
            ("git ls-remote --upload-pack=\"git clone 'ext::ssh localhost id'\" .", _SSH_RULE),
            (
                'git ls-remote --upload-pack="git ls-remote '
                "'ext::ssh -oProxyCommand=ssh% localhost% id far.example.com repo'\" .",
                _SSH_RULE,
            ),
            ("GIT_SSH_COMMAND='git ls-remote localhost:x' git fetch origin", _RULE),
            ("GIT_PROXY_COMMAND='git ls-remote localhost:x' git fetch git://far/x", _RULE),
            ("git -c core.gitProxy='git ls-remote localhost:x' fetch git://far/x", _RULE),
            ("git clone 'ext::git ls-remote localhost:x'", _RULE),
            ("tar --rsh-command='git ls-remote localhost:x' -cf far:a.tar dir", _RULE),
            ("git ls-remote --upload-pack='git ls-remote far.example.com:x' .", None),
            ("git ls-remote --upload-pack='git-upload-pack' .", None),
            (
                'git ls-remote --upload-pack="git ls-remote '
                "'ext::ssh -oProxyCommand=ssh% bastion.example.com far.example.com repo'\" .",
                None,
            ),
            ("GIT_SSH_COMMAND='git ls-remote far.example.com:x' git fetch origin", None),
        ],
        ids=[
            "upload-pack-git",
            "upload-pack-ssh",
            "receive-pack-git",
            "short-u-git",
            "archive-exec-git",
            "upload-pack-git-ssh-scheme",
            "upload-pack-bracketed",
            "upload-pack-encoded-option-value",
            "upload-pack-wraps-ext",
            "upload-pack-wraps-ext-escaped-spaces",
            "ssh-command-env-git",
            "proxy-command-env-git",
            "git-proxy-config-git",
            "ext-git",
            "tar-rsh-command-git",
            "upload-pack-remote-control",
            "upload-pack-program-control",
            "upload-pack-wraps-ext-remote-control",
            "ssh-command-env-remote-control",
        ],
    )
    def test_command_values_reenter_every_floor(self, cmd, rule):
        assert security._is_git_tar_to_self(cmd) is (rule is not None), cmd
        assert _denied_by(cmd) == rule, cmd

    def test_command_values_reach_floors_beyond_ssh_and_git(self, monkeypatch):
        # The re-entry is the whole dispatcher, not a hand-picked pair: a floor
        # that refuses the carried command refuses its carrier too.
        cmd = "git ls-remote --upload-pack='probe-marker localhost' ."
        assert not security._is_git_tar_to_self(cmd), cmd
        monkeypatch.setattr(security, "_is_self_restart", lambda text: "probe-marker" in text)
        assert security._is_git_tar_to_self(cmd), cmd

    def test_a_carried_command_no_floor_can_read_is_refused(self, monkeypatch):
        def unreadable(_text, **_kwargs):
            raise RuntimeError("floor exploded")

        cmd = "git ls-remote --upload-pack='localhost run' ."
        assert not security._is_git_tar_to_self(cmd), cmd
        monkeypatch.setattr(security, "_is_self_kill", unreadable)
        assert security._is_git_tar_to_self(cmd), cmd

    @pytest.mark.parametrize(
        ("chain", "alias", "command"),
        [
            ((0, frozenset({"a"}), frozenset()), "a", "git status"),
            ((0, frozenset(), frozenset({"git status"})), None, "git status"),
            ((_argv_floor._FUNCTION_BIND_CAP, frozenset(), frozenset()), None, "git status"),
        ],
        ids=["alias-on-chain", "command-on-chain", "depth-at-limit"],
    )
    def test_reentry_refuses_a_loop_or_overdeep_chain(self, chain, alias, command):
        assert not _argv_floor._reenter_floors(command, alias=alias)
        previous = _argv_floor._REENTRY_CHAIN.get()
        _argv_floor._REENTRY_CHAIN.set(chain)
        try:
            assert _argv_floor._reenter_floors(command, alias=alias)
        finally:
            _argv_floor._REENTRY_CHAIN.set(previous)

    # A carried command is read with the operator's row set, the way a whole
    # command is: a row turned off stays off for the commands git or tar runs,
    # and the rows still on keep refusing.
    @pytest.mark.parametrize(
        ("off", "cmd", "rule"),
        [
            (_SSH_RULE, "GIT_SSH_COMMAND='ssh localhost' git fetch origin", None),
            (_SSH_RULE, "git -c alias.x='!ssh localhost' x", None),
            (_SSH_RULE, "GIT_SSH_COMMAND='ssh localhost' git -c alias.f=fetch f origin", None),
            (
                "self-protection-kill",
                "git ls-remote --upload-pack='pkill -f kirocrew localhost' .",
                None,
            ),
            (_SSH_RULE, "git ls-remote --upload-pack='git ls-remote localhost:x' .", _RULE),
        ],
        ids=["ssh-env", "ssh-shell-alias", "ssh-ordinary-alias", "kill-program", "git-tar-control"],
    )
    def test_a_disabled_row_stays_off_through_reentry(self, off, cmd, rule):
        assert _denied_by(cmd) == (rule or off), cmd
        effective = list(compute_effective_denied(BUILTIN_DENIED_RULES, (off,), False, (), ()))
        verdict = is_denied(cmd, denied_regexes=effective)
        if rule is None:
            assert verdict is None, cmd
        else:
            assert verdict is not None and _rule_pattern(rule) in verdict.splitlines()[0], cmd

    # A hit found by re-entering a carried command is reported under the row that
    # matched it -- its pattern, its structural note, its diagnostic and its audit
    # event -- not under the git/tar row that carried it.  The top-level floors of
    # those rows do not refuse these commands, so the id comes from the re-entry.
    @pytest.mark.parametrize(
        ("cmd", "rule"),
        [
            ("GIT_SSH_COMMAND='ssh localhost' git fetch origin", _SSH_RULE),
            ("git -c alias.x='!ssh localhost' x", _SSH_RULE),
            (
                "git ls-remote --upload-pack='pkill -f kirocrew localhost' .",
                "self-protection-kill",
            ),
            ("git ls-remote --upload-pack='git ls-remote localhost:x' .", _RULE),
        ],
        ids=["ssh-env", "ssh-shell-alias", "kill-program", "git-tar-itself"],
    )
    def test_a_reentered_hit_reports_the_inner_row(self, cmd, rule, monkeypatch):
        lower = cmd.lower()
        assert not security._is_ssh_to_self(lower) and not security._is_self_kill(lower), cmd
        events: "list[str]" = []
        monkeypatch.setattr(
            security, "_emit_deny_event", lambda _tool, pattern, *_a, **_k: events.append(pattern)
        )
        verdict = is_denied(cmd)
        assert verdict is not None, cmd
        assert verdict.splitlines()[0].endswith(_rule_pattern(rule)), cmd
        assert security._SELF_PROTECTION_FLOOR_NOTES[rule] in verdict, cmd
        assert f"rule={rule} component=argv-floor" in verdict, cmd
        assert events == [_rule_pattern(rule)], cmd

    # An attached ``--opt=value`` is read exactly like the detached ``--opt value``.
    @pytest.mark.parametrize(
        ("template", "value", "denied"),
        [
            ("git push {opt} feat", "localhost:x", True),
            ("git push {opt} feat", "far.example.com:x", False),
            ("git ls-remote {opt} .", "git ls-remote localhost:x", True),
            ("git ls-remote {opt} .", "git ls-remote far.example.com:x", False),
            ("git send-pack {opt} far:x", "git ls-remote localhost:x", True),
            ("git archive {opt} HEAD", "localhost:x", True),
            ("git archive {opt} HEAD", "github.com:x/y", False),
            ("git clone {opt} https://github.com/x/y", "localhost:x", True),
            ("git clone {opt} https://github.com/x/y", "/srv/ref", False),
        ],
        ids=[
            "repo-self",
            "repo-remote",
            "upload-pack-self",
            "upload-pack-remote",
            "receive-pack-self",
            "remote-self",
            "remote-remote",
            "reference-self",
            "reference-local",
        ],
    )
    @pytest.mark.parametrize("form", ["attached", "detached"])
    def test_attached_option_value_reads_like_detached(self, template, value, denied, form):
        name = {
            "git push": "--repo",
            "git ls-remote": "--upload-pack",
            "git send-pack": "--receive-pack",
            "git archive": "--remote",
            "git clone": "--reference",
        }[" ".join(template.split()[:2])]
        quoted = "'" + value + "'"
        opt = f"{name}={quoted}" if form == "attached" else f"{name} {quoted}"
        cmd = template.format(opt=opt)
        assert security._is_git_tar_to_self(cmd) is denied, cmd
