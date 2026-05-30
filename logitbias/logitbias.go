// Package logitbias parses the user-facing logit-bias specification string
// ("<tokenID>:<bias>[,<tokenID>:<bias>...]") into token/bias pairs. It is
// cgo-free so it can be unit-tested without building the C binding.
package logitbias

import (
	"fmt"
	"strconv"
	"strings"
)

// Entry is one parsed token-bias pair.
type Entry struct {
	Token int32
	Bias  float32
}

// Parse converts "id:bias,id:bias" into entries. Whitespace around ids and
// biases is tolerated and empty segments (e.g. a trailing comma) are skipped.
// Empty or whitespace-only input yields (nil, nil). A malformed pair returns a
// non-nil error naming the offending segment. When the same token appears more
// than once the later value wins (matching llama.cpp's logit-bias semantics).
func Parse(s string) ([]Entry, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}

	index := make(map[int32]int)
	entries := make([]Entry, 0)

	for seg := range strings.SplitSeq(s, ",") {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}

		tokStr, biasStr, ok := strings.Cut(seg, ":")
		if !ok {
			return nil, fmt.Errorf("logit_bias: segment %q is not <token>:<bias>", seg)
		}

		tok, err := strconv.ParseInt(strings.TrimSpace(tokStr), 10, 32)
		if err != nil {
			return nil, fmt.Errorf("logit_bias: bad token in %q: %w", seg, err)
		}

		bias, err := strconv.ParseFloat(strings.TrimSpace(biasStr), 32)
		if err != nil {
			return nil, fmt.Errorf("logit_bias: bad bias in %q: %w", seg, err)
		}

		e := Entry{Token: int32(tok), Bias: float32(bias)}
		if pos, seen := index[e.Token]; seen {
			entries[pos] = e

			continue
		}

		index[e.Token] = len(entries)
		entries = append(entries, e)
	}

	if len(entries) == 0 {
		return nil, nil
	}

	return entries, nil
}
