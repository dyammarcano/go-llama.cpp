// Derived from github.com/ollama/ollama/runner/common/stop.go (MIT License).
// Adapted for github.com/dyammarcano/go-llama.cpp.

package streamfilter

import "strings"

// Filter incrementally filters a stream of decoded token pieces: it holds back
// text that might be part of a stop sequence or an incomplete UTF-8 character,
// emits only text that is safe to show, and reports when a stop sequence is hit.
type Filter struct {
	stops   []string
	pending []string
}

// New returns a Filter for the given stop sequences (nil or empty means no
// stops — the Filter then only guards against split UTF-8 characters).
func New(stops []string) *Filter {
	return &Filter{stops: stops}
}

// Push feeds one decoded piece and returns the text now safe to emit (possibly
// empty) plus whether a stop sequence was reached. When stop is true, the
// matched stop sequence and everything after it are dropped.
func (f *Filter) Push(piece string) (emit string, stop bool) {
	f.pending = append(f.pending, piece)
	seq := strings.Join(f.pending, "")

	if found, matched := FindStop(seq, f.stops); found {
		truncated, _ := TruncateStop(f.pending, matched)
		f.pending = nil

		return strings.Join(truncated, ""), true
	}

	if ContainsStopSuffix(seq, f.stops) || IncompleteUnicode(seq) {
		return "", false
	}

	f.pending = nil

	return seq, false
}

// Flush returns any buffered remainder at end-of-generation and clears the
// buffer.
func (f *Filter) Flush() string {
	out := strings.Join(f.pending, "")
	f.pending = nil

	return out
}
