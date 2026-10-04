//go:build darwin || dragonfly || freebsd || illumos || linux || netbsd || openbsd

package engine

import (
	"errors"
	"os"
	"syscall"
)

func lockCacheFile(file *os.File) error {
	for {
		err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX)
		if !errors.Is(err, syscall.EINTR) {
			return err
		}
	}
}
