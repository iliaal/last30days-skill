"""Execute the skill's save-directory setup against isolated configuration."""

from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "last30days"
ENGINE = SKILL / "scripts" / "last30days.py"
BASH = shutil.which("bash")


@pytest.fixture
def shell_env(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    return {
        "PATH": str(bin_dir),
        "HOME": str(Path.home()),
        "LAST30DAYS_PYTHON": sys.executable,
        "SKILL_DIR": str(SKILL),
        "LAST30DAYS_CONFIG_DIR": str(config_dir),
        "LAST30DAYS_CACHE_DIR": str(tmp_path / "cache"),
    }


def write_config(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o600)


@pytest.mark.parametrize(
    ("global_value", "process_value", "expected"),
    [
        ("/tmp/client research", None, "/tmp/client research"),
        ("/tmp/global", "/tmp/process research", "/tmp/process research"),
        ("/tmp/global", "", ""),
        ("", None, ""),
        (None, None, str(Path.home() / "Documents" / "Last30Days")),
        ("/tmp/configured research", "${user_config.memory_dir}", "/tmp/configured research"),
        ("", "${user_config.memory_dir}", ""),
        (None, "${user_config.memory_dir}", str(Path.home() / "Documents" / "Last30Days")),
        ("${user_config.memory_dir}", None, str(Path.home() / "Documents" / "Last30Days")),
        ("${user_config.memory_dir}", "", ""),
        ("/tmp/configured research", "  ${user_config.memory_dir}  ", "/tmp/configured research"),
        (None, "/tmp/${user_config.memory_dir}/research", "/tmp/${user_config.memory_dir}/research"),
    ],
)
def test_documented_memory_resolution(global_value, process_value, expected, shell_env, tmp_path):
    if global_value is not None:
        write_config(
            Path(shell_env["LAST30DAYS_CONFIG_DIR"]) / ".env",
            f'LAST30DAYS_MEMORY_DIR="{global_value}"\n',
        )
    if process_value is not None:
        shell_env["LAST30DAYS_MEMORY_DIR"] = process_value
    assignments = re.findall(r"^LAST30DAYS_MEMORY_DIR=.*$", (SKILL / "SKILL.md").read_text(), re.M)
    assert assignments
    for assignment in set(assignments):
        result = subprocess.run(
            [BASH, "-c", assignment + '\nprintf "%s" "$LAST30DAYS_MEMORY_DIR"'],
            env=shell_env, cwd=tmp_path, text=True, capture_output=True, check=True,
        )
        assert result.stdout == expected


@pytest.mark.parametrize("trust", [None, "1", "0", "global"])
def test_resolver_honors_project_trust(shell_env, tmp_path, trust):
    config_text = "LAST30DAYS_MEMORY_DIR=/tmp/global research\n"
    if trust == "global":
        config_text += "LAST30DAYS_TRUST_PROJECT_CONFIG=1\n"
    elif trust is not None:
        shell_env["LAST30DAYS_TRUST_PROJECT_CONFIG"] = trust
    write_config(Path(shell_env["LAST30DAYS_CONFIG_DIR"]) / ".env", config_text)
    write_config(
        tmp_path / ".claude" / "last30days.env",
        "LAST30DAYS_MEMORY_DIR=project research\nLAST30DAYS_TRUST_PROJECT_CONFIG=1\n",
    )
    result = subprocess.run(
        [sys.executable, str(ENGINE), "--resolve-save-dir"],
        env=shell_env, cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    expected = str(tmp_path / "project research") if trust in {"1", "global"} else "/tmp/global research"
    assert result.stdout == expected + "\n"


@pytest.mark.parametrize("project_value", ["project research", ""])
def test_process_placeholder_uses_trusted_project_setting(shell_env, tmp_path, project_value):
    write_config(
        Path(shell_env["LAST30DAYS_CONFIG_DIR"]) / ".env",
        "LAST30DAYS_MEMORY_DIR=/tmp/global research\nLAST30DAYS_TRUST_PROJECT_CONFIG=1\n",
    )
    write_config(tmp_path / ".claude" / "last30days.env", f"LAST30DAYS_MEMORY_DIR={project_value}\n")
    shell_env["LAST30DAYS_MEMORY_DIR"] = "${user_config.memory_dir}"
    result = subprocess.run(
        [sys.executable, str(ENGINE), "--resolve-save-dir"],
        env=shell_env, cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert result.stdout == (str(tmp_path / project_value) if project_value else "") + "\n"


@pytest.mark.parametrize("flag", ["flag research", "", "${user_config.memory_dir}", "prefix-${user_config.memory_dir}"])
def test_resolver_explicit_flag_wins(shell_env, tmp_path, flag):
    shell_env["LAST30DAYS_MEMORY_DIR"] = "/tmp/process research"
    result = subprocess.run(
        [sys.executable, str(ENGINE), "--resolve-save-dir", "--save-dir", flag],
        env=shell_env, cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert result.stdout == (str(tmp_path / flag) if flag else "") + "\n"


def test_dotenv_path_is_data_not_shell_code(shell_env, tmp_path):
    marker = tmp_path / "must-not-execute"
    value = f"$(printf unsafe > {marker})"
    write_config(Path(shell_env["LAST30DAYS_CONFIG_DIR"]) / ".env", f"LAST30DAYS_MEMORY_DIR={value}\n")
    assignment = re.findall(r"^LAST30DAYS_MEMORY_DIR=.*$", (SKILL / "SKILL.md").read_text(), re.M)[0]
    result = subprocess.run(
        [BASH, "-c", assignment + '\nprintf "%s" "$LAST30DAYS_MEMORY_DIR"'],
        env=shell_env, cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert result.stdout == str(tmp_path / value)
    assert not marker.exists()


def test_resolver_exits_before_auth_or_research(monkeypatch, capsys):
    import last30days
    from lib import env

    assert Path(last30days.__file__).resolve() == ENGINE
    assert Path(env.__file__).resolve() == SKILL / "scripts" / "lib" / "env.py"

    def forbidden(*args, **kwargs):
        pytest.fail("path query entered credential/research configuration")

    monkeypatch.setattr(env, "_load_keychain", forbidden)
    monkeypatch.setattr(env, "_load_pass", forbidden)
    monkeypatch.setattr(env, "_discover_and_apply_x_credentials", forbidden)
    monkeypatch.setattr(sys, "argv", [str(ENGINE), "--resolve-save-dir", "--save-dir", ""])
    assert last30days.main() == 0
    assert capsys.readouterr().out == "\n"


def test_documented_discovery_legs_reuse_captured_directory(shell_env, tmp_path):
    text = (SKILL / "SKILL.md").read_text()
    starts = re.findall(r"^LAST30DAYS_MEMORY_DIR=.*$", text, re.M)
    guards = re.findall(r'^: "\$\{LAST30DAYS_MEMORY_DIR\?.*$', text, re.M)
    assert len(guards) == 2
    config_path = Path(shell_env["LAST30DAYS_CONFIG_DIR"]) / ".env"
    captured = tmp_path / "original research"
    write_config(config_path, f"LAST30DAYS_MEMORY_DIR={captured}\n")
    changed = tmp_path / "changed research"
    command = "\n".join([
        starts[0],
        f"printf '%s\\n' {shlex.quote(f'LAST30DAYS_MEMORY_DIR={changed}')} > {shlex.quote(str(config_path))}",
        *guards,
        'printf "%s" "$LAST30DAYS_MEMORY_DIR"',
    ])
    result = subprocess.run(
        [BASH, "-c", command], env=shell_env, cwd=tmp_path,
        text=True, capture_output=True, check=True,
    )
    assert result.stdout == str(captured)


def test_mock_research_saves_to_documented_dotenv_directory(shell_env, tmp_path):
    target = tmp_path / "client research"
    write_config(Path(shell_env["LAST30DAYS_CONFIG_DIR"]) / ".env", f"LAST30DAYS_MEMORY_DIR={target}\n")
    text = (SKILL / "SKILL.md").read_text()
    assignment = re.findall(r"^LAST30DAYS_MEMORY_DIR=.*$", text, re.M)[0]
    command = assignment + '\n"$LAST30DAYS_PYTHON" "$SKILL_DIR/scripts/last30days.py" OpenAI --mock --quick --no-browser-cookies --emit=md --save-dir="$LAST30DAYS_MEMORY_DIR"'
    result = subprocess.run(
        [BASH, "-c", command], env=shell_env, cwd=tmp_path,
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    artifacts = list(target.glob("*.md"))
    assert artifacts
    assert "OpenAI" in artifacts[0].read_text()


def test_empty_project_setting_disables_bare_engine_save(shell_env, tmp_path):
    target = tmp_path / "global research"
    write_config(
        Path(shell_env["LAST30DAYS_CONFIG_DIR"]) / ".env",
        f"LAST30DAYS_MEMORY_DIR={target}\nLAST30DAYS_TRUST_PROJECT_CONFIG=1\n",
    )
    write_config(tmp_path / ".claude" / "last30days.env", "LAST30DAYS_MEMORY_DIR=\n")
    result = subprocess.run(
        [sys.executable, str(ENGINE), "OpenAI", "--mock", "--quick", "--no-browser-cookies", "--emit=md"],
        env=shell_env, cwd=tmp_path, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OpenAI" in result.stdout
    assert not target.exists()
    assert "Saved output to" not in result.stderr


@pytest.mark.parametrize("disabled", [False, True])
def test_documented_mock_research_placeholder_uses_configured_save_state(shell_env, tmp_path, disabled):
    target = tmp_path / "configured research"
    write_config(
        Path(shell_env["LAST30DAYS_CONFIG_DIR"]) / ".env",
        f"LAST30DAYS_MEMORY_DIR={'' if disabled else target}\n",
    )
    shell_env["LAST30DAYS_MEMORY_DIR"] = "${user_config.memory_dir}"
    assignment = re.findall(r"^LAST30DAYS_MEMORY_DIR=.*$", (SKILL / "SKILL.md").read_text(), re.M)[0]
    command = assignment + '\n"$LAST30DAYS_PYTHON" "$SKILL_DIR/scripts/last30days.py" OpenAI --mock --quick --no-browser-cookies --emit=compact --save-dir="$LAST30DAYS_MEMORY_DIR"'
    result = subprocess.run(
        [BASH, "-c", command], env=shell_env, cwd=tmp_path,
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "✅ All agents reported back!" in result.stdout
    if disabled:
        assert "Raw results saved to" not in result.stdout
        assert not target.exists()
    else:
        artifacts = list(target.glob("*.md"))
        assert len(artifacts) == 1
        assert f"Raw results saved to {artifacts[0]}" in result.stdout
        assert "OpenAI" in artifacts[0].read_text()


def test_skill_footer_claims_only_emitted_saved_paths():
    text = (SKILL / "SKILL.md").read_text()
    law = text.split("**LAW 5 -", 1)[1].split("**LAW 6 -", 1)[0]
    assert "saved-file pointer is optional" in law
    assert "only when emitted" in law
    assert "higher-priority instructions" in law
    footer = text.split("**THEN - Engine footer pass-through", 1)[1].split("**LAST - Invitation", 1)[0]
    assert "only when the engine emitted a saved path" in footer
    assert "Never invent a path" in footer
    assert "higher-priority instructions" in footer
    assert "and ending with `📎 Raw results saved to" not in footer
    assert "└─ 📎 Raw results saved to ..." not in text
    assert "The research script already saved raw data" not in text
