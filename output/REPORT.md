# Ask the Guide: evaluation report

Chunking strategy chosen on dev questions: **section** (threshold 0.01).

| Strategy | Split | Correct answers | Correct refusals | Sent to team leader | Wrong answers | Top-1 hit | Top-3 hit |
|---|---|---|---|---|---|---|---|
| section | dev | 20 | 3 | 0 | 1 | 100% | 100% |
| section | test | 19 | 2 | 0 | 3 | 95% | 95% |
| window | dev | 15 | 3 | 5 | 1 | 85% | 100% |
| window | test | 13 | 4 | 7 | 0 | 85% | 95% |

## Test questions the assistant did not answer correctly

- q27 (wrong_answer): "caller says she is phoning for her dad, email is [EMAIL], can I discuss his account" expected G10, top match G02 at 0.074
- q41 (wrong_answer): "what time does the office canteen close" expected NONE, top match G11 at 0.117
- q46 (wrong_answer): "how many days of annual leave do I get" expected NONE, top match G12 at 0.143

## ROI (assumptions in roi_config.json)

- questions per year: 35200
- answered by assistant share: 0.95
- questions handled per year: 33440
- agent time saved hours: 1672
- team leader time saved hours: 1114
- gross saving gbp: 47373
- running cost gbp: 1312
- wrong answer share: 0.05
- cost of fixing wrong answers gbp: 5866
- net saving gbp: 40194
- build cost gbp: 3000
- payback weeks: 3.9

Sensitivity: with only 1 question per agent per day, net saving is £9,328 and payback is 16.7 weeks.
