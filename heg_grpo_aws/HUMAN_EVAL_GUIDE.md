# Human evaluation: annotator guide

You will label two spreadsheets: `h1_pairs.csv` (60 rows) and `h2_solutions.csv` (up to 80 rows: one row per
lost problem plus one matched retained problem for each; the sheets used in our study had 40 rows, because only 20
problems were lost). Expect about 2–3 hours for the 100 rows of our study (4–5 hours for the full 140); take breaks.

**Rules**
- Work alone. Don't discuss items with the other annotator until you have both finished.
- Don't open `key.json`, and don't try to guess which model wrote a solution.
- Write the label in the `label` column, exactly as shown below.
- `notes` is optional. Use it for anything odd (e.g. "solution truncated", "problem ambiguous").
- If you really cannot decide, use `UNSURE`. Use it rarely; it drops the item from your counts.

---

## H1: same approach or different approach? (`h1_pairs.csv`)

Each row shows a problem and two **correct** solutions to it.

| Label | Meaning |
|---|---|
| `SAME` | Both solutions rely on the same key mathematical idea or method. |
| `DIFFERENT` | They solve the problem in genuinely different ways. |

**Count as DIFFERENT:** a different key step that a teacher would call a different method. For example:
- algebra vs. a geometric argument;
- direct counting vs. complementary counting;
- solving a system by substitution vs. by elimination;
- casework vs. a symmetry argument;
- using a formula vs. deriving the result from first principles.

**Count as SAME:**
- different wording, explanation length or level of detail;
- a different order of the same steps;
- different variable names or notation;
- an extra verification step at the end;
- the same method with slightly different arithmetic.

This follows the "approach-level" definition of Lee et al. (arXiv:2606.29985).

Ask yourself: *if I summarized each solution in one sentence ("set up equations and substitute", "draw the
altitude and use similar triangles"), would the two sentences be the same?*

## H2: is the reasoning valid? (`h2_solutions.csv`)

Each row shows a problem, its reference answer, and **one** model solution whose final boxed answer
matches the reference. Decide whether the solution *earns* that answer.

| Label | Meaning |
|---|---|
| `VALID` | The steps that lead to the answer are mathematically correct. Small slips that don't affect the result are fine (a typo in a sentence, an arithmetic slip that is immediately corrected). |
| `FLAWED` | The answer is right but the reasoning doesn't support it (see below). |

Examples of `FLAWED`:
- a wrong step that happens to land on the right number;
- an unjustified jump straight to the answer;
- contradictory steps, or a computation that gives a different number than the one boxed;
- reasoning that is cut off or abandoned, followed by a guessed answer;
- a "solution" that only restates the problem and then states the answer.

Judge the reasoning as written, not whether you can imagine a fix. You may check arithmetic with a calculator.

---

## After labeling

Return your two CSV files; don't rename them. The organizer runs:

```
python scripts/human_eval.py score
```

It reports agreement between annotators (Cohen's κ). Items you disagree on are left out of the rates.
If a third person adjudicates them, their labels go in `adjudication_h1.csv` / `adjudication_h2.csv`
(columns `item_id,label`) next to `key.json`, and the scoring picks them up.
