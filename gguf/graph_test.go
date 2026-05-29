package gguf

import "testing"

func TestLlamaGraphSize(t *testing.T) {
	// tiny dims chosen so the result is hand-computable:
	// embedding=64 heads=8 embeddingHeads=8 headsKV=8 context=16 batch=8 vocab=128
	full, partial := llamaGraphSize(64, 8, 8, 8, 16, 8, 128)

	// full = max(4*8*(1+4*64+16*(1+8)), 4*8*(64+128))
	//      = max(32*401, 32*192) = max(12832, 6144) = 12832
	if full != 12832 {
		t.Errorf("full = %d, want 12832", full)
	}
	// partial = 4*8*64 + max(
	//   4*8*(1+64+max(16,64)) + 64*64*9/16 + 4*16*(8*8 + 8*8),
	//   4*8*(64+128) + 64*128*105/128)
	// = 2048 + max(32*129 + 2304 + 64*128, 6144 + 6720)
	// = 2048 + max(4128+2304+8192, 12864) = 2048 + max(14624,12864) = 2048+14624 = 16672
	if partial != 16672 {
		t.Errorf("partial = %d, want 16672", partial)
	}
}
