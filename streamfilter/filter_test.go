package streamfilter

import "testing"

// collect drives a Filter through pieces, concatenating emitted text. It stops
// early if a stop is hit, otherwise appends Flush() output.
func collect(f *Filter, pieces ...string) (emitted string, stopped bool) {
	for _, p := range pieces {
		e, s := f.Push(p)
		emitted += e

		if s {
			return emitted, true
		}
	}

	emitted += f.Flush()

	return emitted, false
}

func TestFilterNoStops(t *testing.T) {
	got, stopped := collect(New(nil), "hello ", "world")
	if got != "hello world" || stopped {
		t.Errorf("got %q,%v want 'hello world',false", got, stopped)
	}
}

func TestFilterStopWithinPiece(t *testing.T) {
	got, stopped := collect(New([]string{"<end>"}), "keep<end>drop")
	if got != "keep" || !stopped {
		t.Errorf("got %q,%v want 'keep',true", got, stopped)
	}
}

func TestFilterStopSplitAcrossPieces(t *testing.T) {
	f := New([]string{"<end>"})

	e1, s1 := f.Push("keep<")
	if e1 != "" || s1 {
		t.Errorf("push1 = %q,%v want '',false (held)", e1, s1)
	}

	e2, s2 := f.Push("end>drop")
	if e2 != "keep" || !s2 {
		t.Errorf("push2 = %q,%v want 'keep',true", e2, s2)
	}
}

func TestFilterPartialThatIsNotStop(t *testing.T) {
	f := New([]string{"<end>"})

	e1, _ := f.Push("a<") // held: '<' is a prefix of '<end>'
	e2, _ := f.Push("b")  // '<b' is not a stop prefix -> flush all

	if e1 != "" || e2 != "a<b" {
		t.Errorf("emits = %q,%q want '','a<b'", e1, e2)
	}
}

func TestFilterMultibyteSplit(t *testing.T) {
	f := New(nil)

	e1, _ := f.Push("x" + string([]byte{0xC3})) // first byte of 'é'
	if e1 != "" {
		t.Errorf("push1 = %q want '' (incomplete utf8 held)", e1)
	}

	e2, _ := f.Push(string([]byte{0xA9}) + "y") // second byte + more
	if e2 != "xéy" {
		t.Errorf("push2 = %q want 'xéy'", e2)
	}
}

func TestFilterFlushRemainder(t *testing.T) {
	f := New([]string{"<end>"})

	f.Push("ab<") // held as a possible stop prefix
	if got := f.Flush(); got != "ab<" {
		t.Errorf("Flush = %q want 'ab<'", got)
	}
}
