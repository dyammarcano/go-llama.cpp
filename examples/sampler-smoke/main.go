package main

import (
	"fmt"
	"os"
	"strings"

	llama "github.com/dyammarcano/go-llama.cpp"
)

const defaultModel = "/c/Users/dyamm/.lmstudio/models/lmstudio-community/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf"

func pass(check int, detail string) {
	fmt.Printf("CHECK %d: PASS — %s\n", check, detail)
}

func fail(check int, detail string) {
	fmt.Printf("CHECK %d: FAIL — %s\n", check, detail)
}

func partial(check int, detail string) {
	fmt.Printf("CHECK %d: PARTIAL — %s\n", check, detail)
}

func predict(l *llama.LLama, prompt string, opts ...llama.PredictOption) (string, error) {
	return l.Predict(prompt, opts...)
}

func main() {
	model := defaultModel
	if len(os.Args) > 1 {
		model = os.Args[1]
	}

	fmt.Printf("Loading model: %s\n", model)
	l, err := llama.New(model, llama.SetContext(512), llama.SetGPULayers(0))
	if err != nil {
		fmt.Printf("FATAL: load error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Model loaded OK.")
	fmt.Println()

	prompt := "The capital of France is"

	// -----------------------------------------------------------------------
	// CHECK 1 — greedy determinism
	// -----------------------------------------------------------------------
	fmt.Println("=== CHECK 1: greedy determinism ===")
	greedyOpts := []llama.PredictOption{
		llama.SetTokens(32),
		llama.SetThreads(6),
		llama.SetSeed(1),
		llama.SetTemperature(0),
	}
	out1a, err := predict(l, prompt, greedyOpts...)
	if err != nil {
		fail(1, fmt.Sprintf("first predict error: %v", err))
	} else {
		out1b, err2 := predict(l, prompt, greedyOpts...)
		if err2 != nil {
			fail(1, fmt.Sprintf("second predict error: %v", err2))
		} else {
			fmt.Printf("  Run A: %q\n", out1a)
			fmt.Printf("  Run B: %q\n", out1b)
			switch {
			case out1a == "" || out1b == "":
				fail(1, "one or both outputs are empty")
			case out1a == out1b:
				pass(1, "both outputs identical and non-empty")
			default:
				fail(1, "outputs differ — greedy is non-deterministic")
			}
		}
	}
	fmt.Println()

	// -----------------------------------------------------------------------
	// CHECK 2 — min_p coherent output
	// -----------------------------------------------------------------------
	fmt.Println("=== CHECK 2: min_p coherent ===")
	out2, err := predict(l, prompt,
		llama.SetTokens(32),
		llama.SetThreads(6),
		llama.SetSeed(1),
		llama.SetTemperature(0.8),
		llama.SetMinP(0.05),
	)
	fmt.Printf("  Output: %q\n", out2)
	if err != nil {
		fail(2, fmt.Sprintf("predict error: %v", err))
	} else if out2 == "" {
		fail(2, "output is empty")
	} else {
		pass(2, "non-empty output with min_p=0.05")
	}
	fmt.Println()

	// -----------------------------------------------------------------------
	// CHECK 3 — mirostat v2 + v1 coherent
	// -----------------------------------------------------------------------
	fmt.Println("=== CHECK 3: mirostat v2 ===")
	out3v2, err := predict(l, prompt,
		llama.SetTokens(32),
		llama.SetThreads(6),
		llama.SetSeed(1),
		llama.SetTemperature(0.8),
		llama.SetMirostat(2),
		llama.SetMirostatTAU(5.0),
		llama.SetMirostatETA(0.1),
	)
	fmt.Printf("  Mirostat v2 output: %q\n", out3v2)
	if err != nil {
		fail(3, fmt.Sprintf("mirostat v2 predict error: %v", err))
	} else if out3v2 == "" {
		fail(3, "mirostat v2 output is empty")
	} else {
		pass(3, "mirostat v2 non-empty output")
	}

	fmt.Println("=== CHECK 3b: mirostat v1 ===")
	out3v1, err := predict(l, prompt,
		llama.SetTokens(32),
		llama.SetThreads(6),
		llama.SetSeed(1),
		llama.SetTemperature(0.8),
		llama.SetMirostat(1),
		llama.SetMirostatTAU(5.0),
		llama.SetMirostatETA(0.1),
	)
	fmt.Printf("  Mirostat v1 output: %q\n", out3v1)
	if err != nil {
		fmt.Printf("CHECK 3b: FAIL — mirostat v1 predict error: %v\n", err)
	} else if out3v1 == "" {
		fmt.Println("CHECK 3b: FAIL — mirostat v1 output is empty")
	} else {
		fmt.Printf("CHECK 3b: PASS — mirostat v1 non-empty output\n")
	}
	fmt.Println()

	// -----------------------------------------------------------------------
	// CHECK 4 — logit_bias bans a token
	// -----------------------------------------------------------------------
	fmt.Println("=== CHECK 4: logit_bias token ban ===")

	// Step 1: greedy baseline to find the target word
	out4base, err := predict(l, prompt,
		llama.SetTokens(24),
		llama.SetThreads(6),
		llama.SetSeed(1),
		llama.SetTemperature(0),
	)
	fmt.Printf("  Baseline O1: %q\n", out4base)
	if err != nil {
		fail(4, fmt.Sprintf("baseline predict error: %v", err))
		return
	}

	// Determine the target word (expect "Paris" or similar)
	target := "Paris"
	targetWithSpace := " Paris"
	if !strings.Contains(out4base, target) {
		// Try to find whatever word follows the prompt in the output
		partial(4, fmt.Sprintf("baseline output does not contain %q — logit_bias wiring still tested; O1=%q", target, out4base))
		// Still attempt the TokenizeString + SetLogitBias wiring test
	}

	// Step 2: tokenize the target word to get its ID
	fmt.Printf("  Tokenizing %q ...\n", targetWithSpace)
	nTokens, tokenIDs, tokErr := l.TokenizeString(targetWithSpace,
		llama.SetTokens(24),
		llama.SetThreads(6),
	)
	fmt.Printf("  TokenizeString result: n=%d ids=%v err=%v\n", nTokens, tokenIDs, tokErr)

	if tokErr != nil || nTokens <= 0 || len(tokenIDs) == 0 {
		// Fallback: pick a plausible token ID (common mid-vocab IDs for short words)
		// and still exercise the SetLogitBias call path to show it doesn't crash.
		fallbackID := int32(1)
		if len(tokenIDs) > 0 {
			fallbackID = tokenIDs[0]
		}
		biasStr := fmt.Sprintf("%d:-100", fallbackID)
		fmt.Printf("  TokenizeString unusable (n=%d, err=%v) — using fallback id=%d to test wiring\n", nTokens, tokErr, fallbackID)
		out4banned, err2 := predict(l, prompt,
			llama.SetTokens(24),
			llama.SetThreads(6),
			llama.SetSeed(1),
			llama.SetTemperature(0),
			llama.SetLogitBias(biasStr),
		)
		fmt.Printf("  O2 (fallback ban id=%d): %q\n", fallbackID, out4banned)
		if err2 != nil {
			fail(4, fmt.Sprintf("SetLogitBias call crashed: %v", err2))
		} else {
			partial(4, fmt.Sprintf("TokenizeString unusable; SetLogitBias wiring confirmed (no crash); O1=%q O2=%q", out4base, out4banned))
		}
		return
	}

	// Step 3: ban the token(s) for the target word
	biasParts := make([]string, 0, len(tokenIDs))
	for _, id := range tokenIDs {
		biasParts = append(biasParts, fmt.Sprintf("%d:-100", id))
	}
	biasStr := strings.Join(biasParts, ",")
	fmt.Printf("  Banning token IDs via SetLogitBias(%q)\n", biasStr)

	out4banned, err := predict(l, prompt,
		llama.SetTokens(24),
		llama.SetThreads(6),
		llama.SetSeed(1),
		llama.SetTemperature(0),
		llama.SetLogitBias(biasStr),
	)
	fmt.Printf("  O2 (with ban): %q\n", out4banned)
	if err != nil {
		fail(4, fmt.Sprintf("banned predict error: %v", err))
	} else if strings.Contains(out4base, target) && !strings.Contains(out4banned, target) {
		pass(4, fmt.Sprintf("target word %q absent in O2 — ban took effect; token IDs=%v", target, tokenIDs))
	} else if !strings.Contains(out4base, target) {
		partial(4, fmt.Sprintf("baseline did not contain %q so ban divergence cannot be confirmed; O2=%q; IDs=%v", target, out4banned, tokenIDs))
	} else {
		fail(4, fmt.Sprintf("target word %q still present in O2 — ban had no effect; IDs=%v", target, tokenIDs))
	}
}
