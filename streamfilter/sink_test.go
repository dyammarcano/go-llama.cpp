package streamfilter

import (
	"testing"
	"unicode/utf8"
)

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

func TestSinkPassthrough(t *testing.T) {
	var got []string

	s := NewSink(nil, func(e string) bool { got = append(got, e); return true })
	for _, p := range []string{"Hello", ", ", "world"} {
		if !s.OnToken(p) {
			t.Fatalf("OnToken(%q) = false, want true", p)
		}
	}

	if res := s.Finish(); res != "Hello, world" {
		t.Fatalf("Finish() = %q, want %q", res, "Hello, world")
	}

	if want := []string{"Hello", ", ", "world"}; !equal(got, want) {
		t.Fatalf("emitted = %v, want %v", got, want)
	}
}

func TestSinkStopWithinOnePiece(t *testing.T) {
	s := NewSink([]string{"<end>"}, nil)
	if s.OnToken("hi <end> there") {
		t.Fatalf("OnToken with stop = true, want false (halt)")
	}

	if res := s.Finish(); res != "hi " {
		t.Fatalf("Finish() = %q, want %q", res, "hi ")
	}
}

func TestSinkStopSplitAcrossPieces(t *testing.T) {
	s := NewSink([]string{"<end>"}, nil)
	if !s.OnToken("answer<") {
		t.Fatalf("first OnToken = false, want true (hold)")
	}

	if s.OnToken("end> more") {
		t.Fatalf("second OnToken = true, want false (stop)")
	}

	if res := s.Finish(); res != "answer" {
		t.Fatalf("Finish() = %q, want %q", res, "answer")
	}
}

func TestSinkPartialStopResolvesToText(t *testing.T) {
	s := NewSink([]string{"<end>"}, nil)
	if !s.OnToken("a<") {
		t.Fatalf("first OnToken = false, want true (hold ambiguous tail)")
	}

	if !s.OnToken("x") {
		t.Fatalf("second OnToken = false, want true (disambiguated, not a stop)")
	}

	if res := s.Finish(); res != "a<x" {
		t.Fatalf("Finish() = %q, want %q", res, "a<x")
	}
}

func TestSinkSplitUTF8(t *testing.T) {
	// "é" is bytes 0xC3 0xA9, split across two pieces.
	s := NewSink(nil, nil)
	if !s.OnToken("caf\xc3") {
		t.Fatalf("first OnToken = false, want true (hold incomplete UTF-8)")
	}

	if !s.OnToken("\xa9") {
		t.Fatalf("second OnToken = false, want true")
	}

	res := s.Finish()
	if res != "café" {
		t.Fatalf("Finish() = %q, want %q", res, "café")
	}

	if !utf8.ValidString(res) {
		t.Fatalf("Finish() = %q is not valid UTF-8", res)
	}
}

func TestSinkUserHalt(t *testing.T) {
	s := NewSink(nil, func(e string) bool { return false })
	if s.OnToken("stop me") {
		t.Fatalf("OnToken = true, want false (user halt)")
	}

	if res := s.Finish(); res != "stop me" {
		t.Fatalf("Finish() = %q, want %q", res, "stop me")
	}
}

func TestSinkFlushRemainder(t *testing.T) {
	// A trailing incomplete-UTF-8 byte is held, then surfaced as-is by Finish
	// (matching Filter.Flush at true end-of-generation).
	s := NewSink(nil, nil)
	if !s.OnToken("hi\xc3") {
		t.Fatalf("OnToken = false, want true (hold)")
	}

	if res := s.Finish(); res != "hi\xc3" {
		t.Fatalf("Finish() = %q, want %q", res, "hi\xc3")
	}
}
