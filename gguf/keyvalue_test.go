// Derived from github.com/ollama/ollama/fs/gguf (MIT License).
// Adapted for github.com/dyammarcano/go-llama.cpp.

package gguf

import (
	"reflect"
	"testing"
)

func TestValueScalars(t *testing.T) {
	if got := (Value{int32(7)}).Int(); got != 7 {
		t.Errorf("Int() = %d, want 7", got)
	}

	if got := (Value{uint16(9)}).Uint(); got != 9 {
		t.Errorf("Uint() = %d, want 9", got)
	}

	if got := (Value{float32(1.5)}).Float(); got != 1.5 {
		t.Errorf("Float() = %v, want 1.5", got)
	}

	if got := (Value{true}).Bool(); got != true {
		t.Errorf("Bool() = %v, want true", got)
	}

	if got := (Value{"hi"}).String(); got != "hi" {
		t.Errorf("String() = %q, want hi", got)
	}

	if got := (Value{"hi"}).Int(); got != 0 {
		t.Errorf("Int() on string = %d, want 0", got)
	}
}

func TestValueSlices(t *testing.T) {
	if got := (Value{[]int32{1, 2, 3}}).Ints(); !reflect.DeepEqual(got, []int64{1, 2, 3}) {
		t.Errorf("Ints() = %v, want [1 2 3]", got)
	}

	if got := (Value{[]string{"a", "b"}}).Strings(); !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Errorf("Strings() = %v, want [a b]", got)
	}
}

func TestTensorNumBytes(t *testing.T) {
	ti := TensorInfo{Name: "x", Shape: []uint64{2, 3}, Type: TensorTypeF32}
	if got := ti.NumValues(); got != 6 {
		t.Errorf("NumValues() = %d, want 6", got)
	}

	if got := ti.NumBytes(); got != 24 {
		t.Errorf("NumBytes() = %d, want 24", got)
	}

	q := TensorInfo{Name: "y", Shape: []uint64{32}, Type: TensorTypeQ4_0}
	if got := q.NumBytes(); got != 18 {
		t.Errorf("Q4_0 NumBytes() = %d, want 18", got)
	}
}
