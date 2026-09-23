//go:build !windows

package engine

import (
	"os/exec"
	"syscall"
)

// setProcessGroup puts the engine child in its own process group so a
// timeout kill can reach grandchildren (node bird-search, yt-dlp, grok
// CLI) that the python-side atexit SIGTERM cleanup never reaches — atexit
// does not run on SIGKILL. Mirrors lib/subproc.py's os.setsid discipline.
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killProcessGroup SIGKILLs the whole engine process group (negative pid =
// group). ESRCH and other errors are intentionally ignored: the group may
// already be gone when the deadline fires.
func killProcessGroup(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
}
