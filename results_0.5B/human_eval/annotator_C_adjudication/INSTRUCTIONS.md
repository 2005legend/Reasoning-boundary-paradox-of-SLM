# Third reviewer: is the reasoning valid?

Thank you for helping. This takes about 1.5–2 hours for 25 items. Take breaks.

## What you have
- `h2_solutions.html`: open it in a browser to read the items with the maths typeset.
- `h2_solutions.csv`: open it in Excel or Google Sheets. Type your answer in the `label` column, on the row
  with the same `item_id` as in the HTML.

## The task
Each item shows a maths problem, its reference answer, and **one** solution written by an AI model. The solution's
final boxed answer matches the reference. Decide whether the solution **earns** that answer.

| Label | Meaning |
|---|---|
| `VALID` | The steps that lead to the answer are mathematically correct. Small slips that don't affect the result are fine (a typo in a sentence, an arithmetic slip that is immediately corrected). |
| `FLAWED` | The answer is right but the reasoning doesn't support it. |

Mark it `FLAWED` if:
- a wrong step happens to land on the right number;
- it jumps to the answer without justifying it;
- steps contradict each other, or a computation gives a different number from the one boxed;
- the reasoning is cut off or abandoned, then an answer is guessed;
- it only restates the problem and then states the answer.

Judge the reasoning as written, not whether you can imagine a fix. You may use a calculator.

## Rules
- Work alone, and don't look anything up about where these solutions came from.
- Type exactly `VALID` or `FLAWED` in the `label` column. Use `UNSURE` only if you truly cannot decide.
- `notes` is optional (for example "solution cut off", "problem ambiguous").
- Don't rename the file. Save it as CSV and send it back.
