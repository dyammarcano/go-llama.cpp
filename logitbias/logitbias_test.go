package logitbias

import "testing"

func TestParse(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    []Entry
		wantErr bool
	}{
		{"empty", "", nil, false},
		{"whitespace only", "   ", nil, false},
		{"single", "100:-1.5", []Entry{{Token: 100, Bias: -1.5}}, false},
		{"multiple", "1:2,3:-4", []Entry{{Token: 1, Bias: 2}, {Token: 3, Bias: -4}}, false},
		{"surrounding spaces", " 5 : 0.25 , 6 : -0.5 ", []Entry{{Token: 5, Bias: 0.25}, {Token: 6, Bias: -0.5}}, false},
		{"trailing comma", "7:1,", []Entry{{Token: 7, Bias: 1}}, false},
		{"duplicate last wins", "9:1,9:-100", []Entry{{Token: 9, Bias: -100}}, false},
		{"bad pair no colon", "abc", nil, true},
		{"bad token", "x:1", nil, true},
		{"bad bias", "5:y", nil, true},
		{"empty bias", "5:", nil, true},
		{"empty token", ":1", nil, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Parse(tc.in)
			if (err != nil) != tc.wantErr {
				t.Fatalf("Parse(%q) err = %v, wantErr %v", tc.in, err, tc.wantErr)
			}

			if tc.wantErr {
				return
			}

			if len(got) != len(tc.want) {
				t.Fatalf("Parse(%q) = %v, want %v", tc.in, got, tc.want)
			}

			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("Parse(%q)[%d] = %v, want %v", tc.in, i, got[i], tc.want[i])
				}
			}
		})
	}
}
