package engine

import (
	"os"
	"syscall"
	"unsafe"
)

var lockFileEx = syscall.NewLazyDLL("kernel32.dll").NewProc("LockFileEx")

func lockCacheFile(file *os.File) error {
	const exclusiveLock = 0x00000002
	var overlapped syscall.Overlapped
	result, _, err := lockFileEx.Call(file.Fd(), exclusiveLock, 0, 1, 0, uintptr(unsafe.Pointer(&overlapped)))
	if result == 0 {
		return err
	}
	return nil
}
