// Derived from github.com/ollama/ollama/fs/gguf (MIT License).
// Adapted for github.com/go-skynet/go-llama.cpp.
// Reformatted to satisfy this project's golangci-lint config; logic unchanged from upstream.

package gguf

import (
	"bytes"
	"cmp"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"iter"
	"os"
	"slices"
	"strings"
)

const (
	typeUint8 uint32 = iota
	typeInt8
	typeUint16
	typeInt16
	typeUint32
	typeInt32
	typeFloat32
	typeBool
	typeString
	typeArray
	typeUint64
	typeInt64
	typeFloat64
)

// ErrUnsupported is returned when a GGUF file has an unsupported magic, version, or type.
var ErrUnsupported = errors.New("unsupported")

// File holds an open GGUF file and its lazily-decoded key-value and tensor metadata.
type File struct {
	Magic   [4]byte
	Version uint32

	keyValues *lazy[KeyValue]
	tensors   *lazy[TensorInfo]
	offset    int64

	file   *os.File
	reader *bufferedReader
	bts    []byte
}

// Open opens the GGUF file at path, validates the magic bytes and version,
// and prepares lazy iterators for key-value pairs and tensor metadata.
// The caller must call Close when done.
func Open(path string) (*File, error) {
	f := &File{bts: make([]byte, 4096)}

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	f.file = file
	f.reader = newBufferedReader(f.file, 32<<10)

	if err = binary.Read(f.reader, binary.LittleEndian, &f.Magic); err != nil {
		_ = file.Close()
		return nil, err
	}

	if !bytes.Equal(f.Magic[:], []byte("GGUF")) {
		_ = file.Close()
		return nil, fmt.Errorf("%w: bad magic %q", ErrUnsupported, f.Magic[:])
	}

	if err = binary.Read(f.reader, binary.LittleEndian, &f.Version); err != nil {
		_ = file.Close()
		return nil, err
	}

	if f.Version < 2 {
		_ = file.Close()
		return nil, fmt.Errorf("%w version %v", ErrUnsupported, f.Version)
	}

	f.tensors, err = newLazy(f, f.readTensor)
	if err != nil {
		_ = file.Close()
		return nil, err
	}

	f.tensors.successFunc = func() error {
		offset := f.reader.offset

		alignment := cmp.Or(f.KeyValue("general.alignment").Int(), 32)
		f.offset = offset + (alignment-offset%alignment)%alignment

		return nil
	}

	f.keyValues, err = newLazy(f, f.readKeyValue)
	if err != nil {
		_ = file.Close()
		return nil, err
	}

	return f, nil
}

func (f *File) readTensor() (TensorInfo, error) {
	name, err := readString(f)
	if err != nil {
		return TensorInfo{}, err
	}

	dims, err := read[uint32](f)
	if err != nil {
		return TensorInfo{}, err
	}

	shape := make([]uint64, dims)
	for i := range dims {
		shape[i], err = read[uint64](f)
		if err != nil {
			return TensorInfo{}, err
		}
	}

	type_, err := read[uint32](f)
	if err != nil {
		return TensorInfo{}, err
	}

	offset, err := read[uint64](f)
	if err != nil {
		return TensorInfo{}, err
	}

	return TensorInfo{
		Name:   name,
		Offset: offset,
		Shape:  shape,
		Type:   TensorType(type_),
	}, nil
}

func (f *File) readKeyValue() (KeyValue, error) {
	key, err := readString(f)
	if err != nil {
		return KeyValue{}, err
	}

	t, err := read[uint32](f)
	if err != nil {
		return KeyValue{}, err
	}

	value, err := func() (any, error) {
		switch t {
		case typeUint8:
			return read[uint8](f)
		case typeInt8:
			return read[int8](f)
		case typeUint16:
			return read[uint16](f)
		case typeInt16:
			return read[int16](f)
		case typeUint32:
			return read[uint32](f)
		case typeInt32:
			return read[int32](f)
		case typeUint64:
			return read[uint64](f)
		case typeInt64:
			return read[int64](f)
		case typeFloat32:
			return read[float32](f)
		case typeFloat64:
			return read[float64](f)
		case typeBool:
			return read[bool](f)
		case typeString:
			return readString(f)
		case typeArray:
			return readArray(f)
		default:
			return nil, fmt.Errorf("%w type %d", ErrUnsupported, t)
		}
	}()
	if err != nil {
		return KeyValue{}, err
	}

	return KeyValue{
		Key:   key,
		Value: Value{value},
	}, nil
}

func read[T any](f *File) (t T, err error) {
	err = binary.Read(f.reader, binary.LittleEndian, &t)
	return t, err
}

func readString(f *File) (string, error) {
	n, err := read[uint64](f)
	if err != nil {
		return "", err
	}

	if int(n) > len(f.bts) {
		f.bts = make([]byte, n)
	}

	bts := f.bts[:n]
	if _, err := io.ReadFull(f.reader, bts); err != nil {
		return "", err
	}
	defer clear(bts)

	return string(bts), nil
}

func readArray(f *File) (any, error) {
	t, err := read[uint32](f)
	if err != nil {
		return nil, err
	}

	n, err := read[uint64](f)
	if err != nil {
		return nil, err
	}

	switch t {
	case typeUint8:
		return readArrayData[uint8](f, n)
	case typeInt8:
		return readArrayData[int8](f, n)
	case typeUint16:
		return readArrayData[uint16](f, n)
	case typeInt16:
		return readArrayData[int16](f, n)
	case typeUint32:
		return readArrayData[uint32](f, n)
	case typeInt32:
		return readArrayData[int32](f, n)
	case typeUint64:
		return readArrayData[uint64](f, n)
	case typeInt64:
		return readArrayData[int64](f, n)
	case typeFloat32:
		return readArrayData[float32](f, n)
	case typeFloat64:
		return readArrayData[float64](f, n)
	case typeBool:
		return readArrayData[bool](f, n)
	case typeString:
		return readArrayString(f, n)
	default:
		return nil, fmt.Errorf("%w type %d", ErrUnsupported, t)
	}
}

func readArrayData[T any](f *File, n uint64) (s []T, err error) {
	s = make([]T, n)
	for i := range n {
		e, err := read[T](f)
		if err != nil {
			return nil, err
		}

		s[i] = e
	}

	return s, nil
}

func readArrayString(f *File, n uint64) (s []string, err error) {
	s = make([]string, n)
	for i := range n {
		e, err := readString(f)
		if err != nil {
			return nil, err
		}

		s[i] = e
	}

	return s, nil
}

// Close stops all lazy iterators and closes the underlying file.
func (f *File) Close() error {
	f.keyValues.stop()
	f.tensors.stop()

	return f.file.Close()
}

// KeyValue looks up a metadata key-value pair by key.
// Keys without a namespace prefix are qualified with the model's architecture.
func (f *File) KeyValue(key string) KeyValue {
	if !strings.HasPrefix(key, "general.") && !strings.HasPrefix(key, "tokenizer.") {
		key = f.KeyValue("general.architecture").String() + "." + key
	}

	if index := slices.IndexFunc(f.keyValues.values, func(kv KeyValue) bool {
		return kv.Key == key
	}); index >= 0 {
		return f.keyValues.values[index]
	}

	for keyValue, ok := f.keyValues.next(); ok; keyValue, ok = f.keyValues.next() {
		if keyValue.Key == key {
			return keyValue
		}
	}

	return KeyValue{}
}

// NumKeyValues returns the total number of key-value pairs declared in the file.
func (f *File) NumKeyValues() int {
	return int(f.keyValues.count)
}

// KeyValues returns an iterator over all key-value pairs in the file.
func (f *File) KeyValues() iter.Seq2[int, KeyValue] {
	return f.keyValues.All()
}

// TensorInfo looks up tensor metadata by name, exhausting key-values first if needed.
func (f *File) TensorInfo(name string) TensorInfo {
	if index := slices.IndexFunc(f.tensors.values, func(t TensorInfo) bool {
		return t.Name == name
	}); index >= 0 {
		return f.tensors.values[index]
	}

	// fast-forward through key values if we haven't already
	_ = f.keyValues.rest()
	for tensor, ok := f.tensors.next(); ok; tensor, ok = f.tensors.next() {
		if tensor.Name == name {
			return tensor
		}
	}

	return TensorInfo{}
}

// NumTensors returns the total number of tensors declared in the file.
func (f *File) NumTensors() int {
	return int(f.tensors.count)
}

// TensorInfos returns an iterator over all tensor info entries in the file.
func (f *File) TensorInfos() iter.Seq2[int, TensorInfo] {
	// fast forward through key values if we haven't already
	f.keyValues.rest()

	return f.tensors.All()
}

// TensorReader returns the TensorInfo and an io.Reader for the named tensor's data.
func (f *File) TensorReader(name string) (TensorInfo, io.Reader, error) {
	t := f.TensorInfo(name)
	if t.NumBytes() == 0 {
		return TensorInfo{}, nil, fmt.Errorf("tensor %s not found", name)
	}

	// fast forward through tensor info if we haven't already
	_ = f.tensors.rest()

	return t, io.NewSectionReader(f.file, f.offset+int64(t.Offset), t.NumBytes()), nil
}
