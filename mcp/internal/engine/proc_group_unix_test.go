//go:build !windows

package engine

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// TestSetProcessGroupSetsSetpgid guards the CR-013 fix: the engine child
// must lead its own process group so a timeout kill reaches grandchildren
// (node bird-search, yt-dlp, grok CLI) instead of SIGKILLing only the
// direct python child while its atexit SIGTERM cleanup never runs.
func TestSetProcessGroupSetsSetpgid(t *testing.T) {
	cmd := exec.Command("true")
	setProcessGroup(cmd)
	if cmd.SysProcAttr == nil {
		t.Fatal("SysProcAttr is nil; want Setpgid process-group attribute")
	}
	if !cmd.SysProcAttr.Setpgid {
		t.Fatal("SysProcAttr.Setpgid = false, want true")
	}
}

// TestRunTimeoutKillsGrandchild exercises the group-kill path end to end:
// a stub interpreter spawns a background sleep grandchild, Run hits its
// deadline, and the grandchild must be dead afterwards. With the old
// exec.CommandContext behavior only the direct child died and the
// grandchild kept running.
func TestRunTimeoutKillsGrandchild(t *testing.T) {
	dir := t.TempDir()
	stub := filepath.Join(dir, "python3-group-stub.sh")
	pidFile := filepath.Join(dir, "grandchild.pid")
	script := `#!/usr/bin/env bash
sleep 30 &
echo -n "$!" > "` + pidFile + `"
sleep 30
`
	if err := os.WriteFile(stub, []byte(script), 0o755); err != nil {
		t.Fatalf("write stub: %v", err)
	}
	cache := stageCache(t)

	res, err := Run(context.Background(), RunOptions{
		PythonPath: stub,
		CacheDir:   cache,
		Timeout:    500 * time.Millisecond,
	})
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if !res.TimedOut {
		t.Fatal("TimedOut = false, want true")
	}

	raw, readErr := os.ReadFile(pidFile)
	if readErr != nil {
		t.Fatalf("grandchild pid file missing: %v", readErr)
	}
	pid, convErr := strconv.Atoi(strings.TrimSpace(string(raw)))
	if convErr != nil || pid <= 0 {
		t.Fatalf("bad grandchild pid %q: %v", raw, convErr)
	}

	// SIGKILL delivery is async; poll for the process to disappear.
	deadline := time.Now().Add(5 * time.Second)
	for {
		if kerr := syscall.Kill(pid, 0); kerr != nil {
			if !errors.Is(kerr, syscall.ESRCH) {
				t.Logf("kill(pid, 0) = %v; treating as dead", kerr)
			}
			return
		}
		if time.Now().After(deadline) {
			// Best-effort cleanup so a regression does not leak sleeps.
			_ = syscall.Kill(pid, syscall.SIGKILL)
			t.Fatalf("grandchild pid %d still alive 5s after timeout; group kill failed", pid)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
