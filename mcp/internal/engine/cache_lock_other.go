//go:build !darwin && !dragonfly && !freebsd && !illumos && !linux && !netbsd && !openbsd && !windows

package engine

import (
	"fmt"
	"os"
	"runtime"
)

func lockCacheFile(file *os.File) error {
	return fmt.Errorf("cache locking is unsupported on %s", runtime.GOOS)
}
