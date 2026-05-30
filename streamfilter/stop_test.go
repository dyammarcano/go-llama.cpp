package streamfilter

import (
	"reflect"
	"testing"
)

func TestFindStop(t *testing.T) {
	stops := []string{"</s>", "STOP"}
	if ok, m := FindStop("hello</s>", stops); !ok || m != "</s>" {
		t.Errorf("FindStop hit = %v,%q want true,</s>", ok, m)
	}
	if ok, _ := FindStop("hello world", stops); ok {
		t.Error("FindStop should miss")
	}
	if ok, _ := FindStop("anything", nil); ok {
		t.Error("FindStop with no stops should be false")
	}
}

func TestContainsStopSuffix(t *testing.T) {
	stops := []string{"<end>"}
	if !ContainsStopSuffix("foo<", stops) {
		t.Error("tail '<' is a prefix of '<end>' -> true")
	}
	if !ContainsStopSuffix("foo<en", stops) {
		t.Error("tail '<en' is a prefix of '<end>' -> true")
	}
	if ContainsStopSuffix("foobar", stops) {
		t.Error("no prefix match -> false")
	}
	if ContainsStopSuffix("foo", nil) {
		t.Error("no stops -> false")
	}
}

func TestTruncateStop(t *testing.T) {
	got, trunc := TruncateStop([]string{"ab", "cd"}, "bc")
	if !reflect.DeepEqual(got, []string{"a"}) || !trunc {
		t.Errorf("TruncateStop split = %v,%v want [a],true", got, trunc)
	}
	got, trunc = TruncateStop([]string{"ab", "cd"}, "cd")
	if !reflect.DeepEqual(got, []string{"ab"}) || trunc {
		t.Errorf("TruncateStop boundary = %v,%v want [ab],false", got, trunc)
	}
	got, trunc = TruncateStop([]string{"ab", "cd"}, "zz")
	if !reflect.DeepEqual(got, []string{"ab", "cd"}) || trunc {
		t.Errorf("TruncateStop absent = %v,%v want [ab cd],false", got, trunc)
	}
}

func TestIncompleteUnicode(t *testing.T) {
	if IncompleteUnicode("hello") {
		t.Error("ASCII is complete")
	}
	if IncompleteUnicode("héllo") {
		t.Error("complete multibyte is complete")
	}
	if !IncompleteUnicode("h" + string([]byte{0xC3})) {
		t.Error("lone 0xC3 lead byte -> incomplete")
	}
	if !IncompleteUnicode(string([]byte{0xE2, 0x82})) {
		t.Error("truncated 3-byte sequence -> incomplete")
	}
}
