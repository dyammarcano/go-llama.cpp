package main

import (
	"fmt"
	"os"

	llama "github.com/go-skynet/go-llama.cpp"
)

// Minimal non-interactive generation smoke test for the modernized binding.
// usage: smoketest <model.gguf> [prompt]
func main() {
	if len(os.Args) < 2 {
		fmt.Println("usage: smoketest <model.gguf> [prompt]")
		os.Exit(2)
	}

	model := os.Args[1]

	prompt := "The capital of France is"
	if len(os.Args) > 2 {
		prompt = os.Args[2]
	}

	l, err := llama.New(model, llama.SetContext(512), llama.SetGPULayers(0))
	if err != nil {
		fmt.Println("load error:", err)
		os.Exit(1)
	}

	// Apply the model's chat template when it has one, so instruct/chat models
	// answer the prompt instead of continuing it. Base models (and any model
	// without an embedded template) fall back to the raw prompt.
	input := prompt
	if formatted, terr := l.ApplyChatTemplate("", prompt); terr == nil && formatted != "" {
		input = formatted
	}

	fmt.Printf("PROMPT: %q\nOUTPUT: ", prompt)

	res, err := l.Predict(input,
		llama.SetTokens(24),
		llama.SetThreads(6),
		llama.SetSeed(1),
		llama.SetTopK(40),
		llama.SetTopP(0.95),
		llama.SetTokenCallback(func(t string) bool { fmt.Print(t); return true }),
	)
	if err != nil {
		fmt.Println("\npredict error:", err)
		os.Exit(1)
	}

	fmt.Printf("\n--- FINAL ---\n%s\n", res)
}
