// Derived from github.com/ollama/ollama/runner/common/stop.go (MIT License).
// Adapted for github.com/dyammarcano/go-llama.cpp.

package streamfilter

import "strings"

// FindStop reports whether any stop sequence is a substring of sequence,
// returning the first match.
func FindStop(sequence string, stops []string) (bool, string) {
	for _, stop := range stops {
		if strings.Contains(sequence, stop) {
			return true, stop
		}
	}

	return false, ""
}

// ContainsStopSuffix reports whether the tail of sequence equals a prefix of any
// stop sequence — i.e. sequence may be partway through a stop that completes in
// a later piece.
func ContainsStopSuffix(sequence string, stops []string) bool {
	for _, stop := range stops {
		for i := 1; i <= len(stop); i++ {
			if strings.HasSuffix(sequence, stop[:i]) {
				return true
			}
		}
	}

	return false
}

// TruncateStop removes the provided stop string from pieces, returning the
// partial pieces with stop removed, including truncating the last piece if
// required (and signalling if this was the case).
func TruncateStop(pieces []string, stop string) ([]string, bool) {
	joined := strings.Join(pieces, "")

	index := strings.Index(joined, stop)
	if index == -1 {
		return pieces, false
	}

	joined = joined[:index]

	// Split the truncated string back into pieces of their original lengths.
	lengths := make([]int, len(pieces))
	for i, piece := range pieces {
		lengths[i] = len(piece)
	}

	var result []string

	tokenTruncated := false
	start := 0

	for _, length := range lengths {
		if start >= len(joined) {
			break
		}

		end := start + length
		if end > len(joined) {
			end = len(joined)
			tokenTruncated = true
		}

		result = append(result, joined[start:end])
		start = end
	}

	return result, tokenTruncated
}

// IncompleteUnicode reports whether the trailing bytes of token form an
// incomplete (truncated) multibyte UTF-8 character.
func IncompleteUnicode(token string) bool {
	incomplete := false

	// check if there is an incomplete UTF-8 character at the end
	for i := 1; i < 5 && i <= len(token); i++ {
		c := token[len(token)-i]

		if (c & 0xc0) == 0x80 {
			// continuation byte: 10xxxxxx
			continue
		}

		//nolint:gocritic // faithful lift; the trailing break must target the
		// for loop, so this cannot become a switch (break would exit the switch).
		if (c & 0xe0) == 0xc0 {
			// 2-byte character: 110xxxxx ...
			incomplete = i < 2
		} else if (c & 0xf0) == 0xe0 {
			// 3-byte character: 1110xxxx ...
			incomplete = i < 3
		} else if (c & 0xf8) == 0xf0 {
			// 4-byte character: 11110xxx ...
			incomplete = i < 4
		}

		// else 1-byte character or invalid byte
		break
	}

	return incomplete
}
