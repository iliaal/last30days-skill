package engine

import (
	"crypto/sha256"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// engineSourceDir resolves the git source of truth for the vendored engine
// (skills/last30days/scripts/) from this test file's location, independent
// of the caller's working directory.
func engineSourceDir(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller: cannot locate test file")
	}
	// This file lives in mcp/internal/engine/; the repo root is three levels up.
	dir := filepath.Join(filepath.Dir(thisFile), "..", "..", "..",
		"skills", "last30days", "scripts")
	abs, err := filepath.Abs(dir)
	if err != nil {
		t.Fatalf("resolve engine source dir: %v", err)
	}
	if _, err := os.Stat(filepath.Join(abs, "last30days.py")); err != nil {
		t.Fatalf("engine source not found at %s: %v", abs, err)
	}
	return abs
}

// wantFile reports whether a source-tree relative path is part of what
// scripts/sync-engine.sh mirrors: the last30days.py entry point plus the
// lib/ tree, minus interpreter caches the script strips for determinism.
func wantFile(rel string) bool {
	if rel == "last30days.py" {
		return true
	}
	if rel != "lib" && !strings.HasPrefix(rel, "lib/") {
		return false
	}
	if strings.Contains(rel, "__pycache__") || strings.HasSuffix(rel, ".pyc") {
		return false
	}
	return true
}

func hashFileFS(fsys fs.FS, name string) (string, error) {
	data, err := fs.ReadFile(fsys, name)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return fmt.Sprintf("%x", sum), nil
}

// TestVendoredMatchesSource guards the sync-engine.sh contract: the embedded
// engine must be byte-identical to the git source it mirrors. A developer who
// edits skills/last30days/scripts/ and runs `go test` without re-syncing gets
// a failure naming the fix; CI runs scripts/sync-engine.sh before go test so
// the check also fails loudly if that step ever goes missing or silently
// no-ops (a fresh checkout embeds only the .gitkeep anchor).
func TestVendoredMatchesSource(t *testing.T) {
	srcDir := engineSourceDir(t)

	embedded, err := EngineFS()
	if err != nil {
		t.Fatalf("EngineFS: %v", err)
	}
	const resync = "run `bash scripts/sync-engine.sh` from mcp/ to resync " +
		"(source of truth: skills/last30days/scripts/, see mcp/README.md)"

	embeddedHashes := map[string]string{}
	if err := fs.WalkDir(embedded, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || path == ".gitkeep" {
			return nil
		}
		h, err := hashFileFS(embedded, path)
		if err != nil {
			return err
		}
		embeddedHashes[path] = h
		return nil
	}); err != nil {
		t.Fatalf("walk embedded engine: %v", err)
	}

	sourceHashes := map[string]string{}
	if err := filepath.WalkDir(srcDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(srcDir, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if !wantFile(rel) {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		sum := sha256.Sum256(data)
		sourceHashes[rel] = fmt.Sprintf("%x", sum)
		return nil
	}); err != nil {
		t.Fatalf("walk engine source: %v", err)
	}

	var problems []string
	for rel, want := range sourceHashes {
		got, ok := embeddedHashes[rel]
		if !ok {
			problems = append(problems, "missing from embed: "+rel)
			continue
		}
		if got != want {
			problems = append(problems, "stale content in embed: "+rel)
		}
	}
	for rel := range embeddedHashes {
		if _, ok := sourceHashes[rel]; !ok {
			problems = append(problems, "embed has file with no source: "+rel)
		}
	}
	if len(problems) > 0 {
		t.Fatalf("vendored engine differs from git source (%d file(s)):\n  %s\n%s",
			len(problems), strings.Join(problems, "\n  "), resync)
	}
}

var minPythonRe = regexp.MustCompile(`(?m)^MIN_PYTHON\s*=\s*\((\d+),\s*(\d+)\)`)

// TestMinPythonVersionMatchesEngine keeps the Go duplicate of the engine's
// MIN_PYTHON floor honest without hardcoding the value: the expected version
// is parsed out of skills/last30days/scripts/last30days.py itself, so an
// engine-side bump fails here with the exact file and line to mirror instead
// of drifting silently.
func TestMinPythonVersionMatchesEngine(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(engineSourceDir(t), "last30days.py"))
	if err != nil {
		t.Fatalf("read engine entry point: %v", err)
	}
	m := minPythonRe.FindSubmatch(src)
	if m == nil {
		t.Fatal("MIN_PYTHON = (major, minor) not found in last30days.py; " +
			"update minPythonRe in sync_contract_test.go")
	}
	want := string(m[1]) + "." + string(m[2])
	if MinPythonVersion != want {
		t.Fatalf("MinPythonVersion = %q, engine MIN_PYTHON = %q; "+
			"update the Go constant to match last30days.py",
			MinPythonVersion, want)
	}
}
