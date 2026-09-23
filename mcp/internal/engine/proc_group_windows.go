//go:build windows

package engine

import (
	"os/exec"
)

// setProcessGroup is a no-op on Windows: syscall.SysProcAttr has no Setpgid
// field and process groups are a POSIX concept. The timeout path falls back
// to killing the direct child.
func setProcessGroup(cmd *exec.Cmd) {
}

// killProcessGroup kills the direct child on Windows; there is no
// kill(-pgid) equivalent. Grandchild cleanup there relies on the
// python-side atexit handler, which runs on TerminateProcess-observable
// exits where possible.
func killProcessGroup(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
}
