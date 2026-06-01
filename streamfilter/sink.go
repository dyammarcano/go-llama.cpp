package streamfilter

import "strings"

// Sink routes each decoded token piece through a Filter: it forwards only
// safe-to-emit text to an optional user callback, accumulates the full filtered
// text for the caller's return value, and reports when generation should halt.
//
// Sink has no cgo dependency; the in-process llama binding uses it as the glue
// for its token-callback path, but it is exercised entirely with headless tests.
type Sink struct {
	f    *Filter
	user func(string) bool // optional; may be nil
	buf  strings.Builder
}

// NewSink returns a Sink that filters against stops and forwards emitted text to
// user (which may be nil). Empty/nil stops means no stop sequences — only
// incomplete-UTF-8 hold-back applies.
func NewSink(stops []string, user func(string) bool) *Sink {
	return &Sink{f: New(stops), user: user}
}

// OnToken feeds one decoded piece. It accumulates and forwards the safe-to-emit
// text, and returns false when generation should halt — either because a stop
// sequence was reached or because the user callback returned false.
func (s *Sink) OnToken(piece string) bool {
	emit, stop := s.f.Push(piece)
	if emit != "" {
		s.buf.WriteString(emit)

		if s.user != nil && !s.user(emit) {
			return false
		}
	}

	return !stop
}

// Finish flushes any text held at end-of-generation, forwards it to the user
// callback, and returns the full accumulated filtered text.
func (s *Sink) Finish() string {
	if rem := s.f.Flush(); rem != "" {
		s.buf.WriteString(rem)

		if s.user != nil {
			s.user(rem)
		}
	}

	return s.buf.String()
}
