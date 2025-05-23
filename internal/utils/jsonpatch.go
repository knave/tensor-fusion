package utils

import (
	"strings"
)

// EscapeJSONPointer escapes a string according to the JSON Pointer spec (RFC 6901).
// It escapes '~' as '~0' and '/' as '~1'.
func EscapeJSONPointer(s string) string {
	// Order is important: we must escape ~ first so we don't double-escape
	s = strings.ReplaceAll(s, "~", "~0")
	s = strings.ReplaceAll(s, "/", "~1")
	return s
}
