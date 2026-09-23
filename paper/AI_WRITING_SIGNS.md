# Signs of AI writing: a checklist for editing the paper

Compiled 2026-10-05 from:
- [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) (WP:AISIGNS), the
  editor guide Wikipedia uses to spot AI text. The full page source was read.
- Studies of AI vocabulary in scientific papers:
  - [Kobak et al., *Science Advances* 2025](https://www.science.org/doi/10.1126/sciadv.adt3813)
    ([arXiv:2406.07016](https://arxiv.org/abs/2406.07016));
  - [Liang et al. 2024, "Mapping the increasing use of LLMs in scientific papers"](https://arxiv.org/abs/2404.01268);
  - [Liang et al. 2024, AI-modified peer reviews](https://arxiv.org/abs/2403.07183).
- How detectors work: [GPTZero](https://gptzero.me/news/how-ai-detectors-work/).
- Detector bias: [Liang et al. 2023, *Patterns*](https://www.cell.com/patterns/fulltext/S2666-3899(23)00130-7).
- IEEE policy: [Author Guidelines for AI-Generated Text](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/)
  and [IEEE conference submission policies](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/).

---

## 0. Read this first

1. **IEEE requires disclosure.** Any AI-generated content (text, figures, code) must be disclosed in the
   **acknowledgments**. The disclosure names the AI system, the sections it touched and how much it did. Using AI
   only for editing or grammar is "generally outside the intent" of the rule, but disclosure is still recommended.
   Authors stay fully responsible for the content. An undisclosed AI-drafted paper can be rejected or retracted
   whatever a detector says. Rewriting in your own words is right; hiding the AI help is not.
2. **No method guarantees "0% AI".**
   - Detectors have "non-trivial error rates" (WP:AISIGNS), and paraphrasing, formatting or a model they weren't
     trained on can fool them.
   - They also flag real human writing. 7 detectors labelled **61.3%** of essays written by non-native English
     speakers as AI on average, and at least one detector flagged **97.8%** of them. US students' essays were
     judged almost perfectly (Liang et al. 2023).
   - The reason: detectors score *predictable* word choice as AI-like.
3. **What detectors measure.**
   - *Perplexity*: how predictable each next word is; AI picks the most likely word.
   - *Burstiness*: how much sentence length and structure vary; AI is evenly paced.
   - Modern detectors add trained classifiers on top of these (GPTZero).
4. **Humans are bad at this too.** Wikipedia cites a 2025 study where people were no better than chance, and
   experts who use LLMs a lot reached 90%. **Signs work in combination; one alone proves nothing.**

The rest of this file lists each sign: what it looks like, the giveaway words, and the plain fix.

---

## 1. Content signs (*what* is said)

AI regresses to the mean: it swaps specific, unusual facts for generic, important-sounding statements.
"Inventor of the first train-coupling device" becomes "a revolutionary titan of industry". The text gets
*less specific and more exaggerated* at the same time.

### 1.1 Puffed-up significance, legacy and "broader trends"
- **Looks like:** the topic "represents a shift", "contributes to the broader…", "sparks debates about…".
- **Watch:** *stands/serves as, is a testament/reminder, a crucial/pivotal/vital/significant/key role/moment,
  underscores/highlights its importance, reflects broader, symbolizing its ongoing/enduring/lasting,
  contributing to the, setting the stage for, marking/shaping the, represents/marks a shift, key turning point,
  evolving landscape, focal point, indelible mark, deeply rooted.*
- **Fix:** state the concrete fact. Keep "why it matters" only if you have evidence, and then cite it.

### 1.2 Superficial "-ing" analysis tacked onto sentences
- **Looks like:** "…in 2019, *highlighting the growing importance of X*."
- **Watch:** *highlighting / underscoring / emphasizing / ensuring / reflecting / symbolizing / contributing to /
  fostering / cultivating / encompassing / enhancing …, valuable insights, align/resonate with.*
- **Fix:** delete the trailing clause, or turn it into its own claim backed by a number or a citation.

### 1.3 Promotional, advert-like tone
- **Watch:** *boasts a, vibrant, rich, profound, enhancing, showcasing, exemplifies, commitment to, nestled, in the
  heart of, groundbreaking, renowned, featuring, diverse array.*
- **Fix:** neutral, measurable wording. "Our method improves Pass@32 by 1.4 pp", not "a groundbreaking method".

### 1.4 Vague "connection" instead of a direct relation
- **Watch:** *in connection with, connected to, in association with, associated with.*
- **Fix:** say the actual relation: "X causes Y", "X is a special case of Y", "X uses Y".

### 1.5 Vague attributions and overgeneralization (weasel words)
- **Watch:** *Experts argue, Observers have cited, Some critics argue, Industry reports, several studies* (when
  you cite one), *such as* (before a list that is actually complete).
- **Fix:** name who said it, cite exactly them, and don't inflate one source into "many".

### 1.6 Canned emphasis on coverage or notability
- **Watch:** *independent coverage, featured/profiled in, leading expert, active social media presence.*
- **Fix:** rarely relevant to a paper. Just cite.

### 1.7 Formulaic "Challenges / Future prospects" ending
- **Looks like:** "Despite its [praise], X faces several challenges…", followed by a vague hopeful close.
- **Watch:** *Despite these challenges, Challenges and Legacy, Future Outlook.*
- **Fix:** give specific limitations ("3 seeds", "one model") and specific next experiments.

## 2. Language and grammar signs (*how* it is said)

### 2.1 "AI vocabulary" (one of the strongest tells when many co-occur)

Wikipedia's lists, by era:
- **2023 to mid-2024 (GPT-4):** *additionally, boasts, bolstered, crucial, delve, emphasizing, enduring, garner,
  intricate/intricacies, interplay, key, landscape, meticulous(ly), pivotal, underscore, tapestry, testament,
  valuable, vibrant.*
- **Mid-2024 to mid-2025 (GPT-4o):** *align with, bolstered, crucial, emphasizing, enhance, enduring, fostering,
  highlighting, pivotal, showcasing, underscore, vibrant.*
- **Mid-2025 onward (GPT-5):** *emphasizing, enhance, highlighting, showcasing* (plus the coverage words in 1.6).
- **Grok** overuses pseudo-scientific words: *causal, empirical, correlate, underscore.*
- The overall watch list adds *deep dive, highlight* (as a verb), *key* (as an adjective), *landscape* (abstract),
  and *robust*.

**In scientific papers specifically:**
- Kobak et al. (15M PubMed abstracts) estimate that at least **13.5%** of 2024 abstracts were LLM-processed, and
  up to 40% in some fields.
  - Strongest excess words: *delves* (28× expected), *underscores* (13.8×), *showcasing* (10.7×).
  - Frequent ones: *potential, findings, crucial.*
  - Their 10-word marker set: *across, additionally, comprehensive, crucial, enhancing, exhibited, insights,
    notably, particularly, within.*
  - Also in their list: *intricate, meticulous(ly), pivotal, realm, nuanced, multifaceted, noteworthy, invaluable,
    groundbreaking, transformative, seamless, leverages, utilizes, facilitates, harness, elucidate, unveil(ing),
    unraveling, paving, poised, burgeoning, uncharted, underexplored, commendable, compelling, remarkable,
    unparalleled, interplay, interconnectedness, endeavors, strategically, swiftly.*
- Liang et al. (950k papers) found up to **17.5%** of CS abstracts LLM-modified by Feb 2024.
- In AI-modified peer reviews, *commendable* (9.8×), *meticulous* (34.7×) and *intricate* (11.2×) jumped.

**Fix:**
- Use the plain word: *shows* (not showcases), *important* or nothing (not crucial/pivotal), *study* (not delve
  into), *uses* (not leverages/utilizes).
- Note: only those exact words are overused, not their synonyms. Ordinary formal or academic vocabulary is
  **not** a sign.

### 2.2 Avoiding plain "is / are / has"
- **Watch:** *serves as, stands as, marks, functions as, operates as, represents, boasts, features, maintains,
  offers, refers to.*
- One study found a **>10% drop in "is"/"are" in academic writing in 2023**. AI "copyedits" make this worse.
- **Fix:** write "X is Y" and "X has Y".

### 2.3 Negative parallelisms
- **Watch:**
  - *Not only … but (also) …*
  - *It's not X, it's Y* / *no X, no Y, just Z*
  - *Y rather than X* (the reversed form; Wikipedia says it also appears in Claude output)
- **Fix:** just state Y. Keep a contrast only when a reader would really expect X.

### 2.4 Rule of three
- **Looks like:** "adjective, adjective, adjective"; "short phrase, short phrase, and short phrase".
- **Fix:** list as many items as you really have: one, two or four.

### 2.5 Elegant variation (synonym cycling)
- Older models avoided repeating a word, so one thing gets three names.
- **Fix:** in a paper, **always use the same term for the same thing**. Repetition is clarity.

### 2.6 Stiff synonyms and missing everyday constructions

Wikipedia's "signs of human writing" are things humans do *more* than AI:
- simple *there is / it has*;
- plain verbs: *wrote* not *authored*, *used* not *utilized*, *tried* not *attempted*, *moved* not *relocated*;
- definite statements: *the first, the only, one of the best*;
- hedges and intensifiers: *very, perhaps, tends to*;
- slightly wordy everyday phrases: *as a result of, in order to, the fact that*.

**Fix:** let these back in where they are natural.

### 2.7 Formulaic transitions (weak sign alone)
- **Watch:** sentence-initial *Additionally, Moreover, Furthermore, Notably, Consequently.*
- **Fix:** drop most of them. The logic should carry the link.

### 2.8 Didactic disclaimers and section summaries (older models)
- **Watch:** *It's important / crucial to note, worth noting, may vary; In summary, In conclusion, Overall*, and
  paragraphs that end by restating themselves.
- **Fix:** delete them. In a paper, keep one Conclusion section and no recap at the end of every paragraph.

### 2.9 Uniform rhythm (what "burstiness" detectors pick up)
- Same-length sentences, same paragraph shape, every paragraph with topic sentence → explanation → wrap-up.
- **Fix:** vary naturally. Write a short sentence after a long one, and don't force symmetry.

## 3. Style and formatting signs

| Sign | Looks like | Fix |
|---|---|---|
| Title Case Headings | every main word capitalized | follow the venue style (IEEE templates handle sections) |
| Boldface overuse | bold on every key phrase | almost none in a paper |
| Inline-header vertical lists | `• **Term:** explanation` bullet after bullet | turn into prose, or plain lists without the bold mini-headings |
| Em-dash overuse | dashes, often with spaces around them, where commas/colons/parentheses fit | commas or parentheses; at most a few true em dashes (no spaces) |
| Emoji as bullets or headings | 📌 ✅ | never in a paper |
| Tiny tables | 2-row tables that could be a sentence | a sentence |
| Curly vs straight quotes mixed | “…” next to "…" | consistent; in LaTeX use ``…'' |
| Skipped heading levels, extra thematic breaks | | normal structure |
| Wrong variety of English | e.g. American spelling from a writer who normally uses British/Indian spelling | use the variety you normally write in, consistently |

## 4. Chatbot leftovers (instant giveaways)

- **Talking to the user:** *Certainly!, Of course!, I hope this helps, Let me know…, Would you like…, Here is a…*
- **Knowledge-cutoff talk:** *as of my last training update.*
- **Refusals:** *as an AI language model, I'm sorry, but…*
- **Placeholders left in:** *[Insert name], [Your University]*.
- **"Source not available" hedges:** *While specific details are limited…, not widely documented…, based on
  available information…*
- **Markup artifacts** pasted from chat:
  - Markdown `**bold**` / `##`;
  - ChatGPT's `:contentReference[oaicite:0]`, `turn0search0`, `utm_source=chatgpt.com` in URLs;
  - Gemini's `[cite: 1]`;
  - DeepSeek's 【】 brackets and † daggers;
  - Grok's `grok_card`;
  - Perplexity's `ppl-ai-file-upload`.
- **Abrupt cut-offs** mid-sentence.

## 5. Citation signs (the most damaging in a paper)

- **Hallucinated references:**
  - DOIs that don't resolve, or that resolve to an unrelated paper;
  - invalid ISBNs;
  - real books cited without page numbers, or with pages that don't support the claim;
  - authors or years that don't match the record.
- **Citations that don't support the sentence**: the source exists but says something else.
- **Fix:** verify every entry against the actual record, and re-read the cited passage for every claim. (All
  41 entries in `refs.bib` were generated from arXiv's own records, so this is done for identity; the
  *does it support the claim* check still needs a human pass.)

## 6. Document-level signs

- **Sudden style shift.** Flawless prose next to the author's normal writing (emails, earlier reports). Your
  thesis, your earlier documents and this paper should sound like the same person.
- **Can you explain every choice?** Wikipedia's strongest human signal is being able to explain why a sentence or
  choice is there. Be ready to defend every claim in a viva or a review.

## 7. NOT reliable signs (don't over-correct)

From WP:AISIGNS, "Ineffective indicators":
- perfect grammar;
- a mix of casual and formal registers;
- "bland" prose;
- formal or academic vocabulary *in general*;
- transition words on their own;
- unsourced content.

Don't make the paper worse trying to dodge these, for example by adding mistakes or slang.

---

## 8. Where our paper stands (`main.tex`, scanned 2026-10-05)

**Vocabulary: clean.**
- None of *delve, crucial, pivotal, showcase, underscore, highlight, landscape, robust, comprehensive, notably,
  insights, additionally, furthermore, moreover, leverage, utilize* appear.
- *within* ×7 and *across* ×3 are on Kobak's list, but here they are technical terms ("within a rollout group",
  "across prompts"). Keep them.

**Structure: these are the real tells left.**

| Tell | Count / example | Suggested fix |
|---|---|---|
| Semicolons | 56 | split most into separate sentences |
| Colons | 66 | many "X: Y" explanations; rephrase about half as plain sentences |
| "rather than" contrasts (2.3) | ×5 | keep the 1–2 that really contrast; state the rest directly |
| Inline-header list (3) | the contribution bullets start with an italic mini-question | fine in ML papers, but consider plain prose bullets |
| Rule of three (2.4) | e.g. "is it real, what does it lose, do they compose" | here it matches three real questions, so keep it; check other triples |
| Uniform, dense paragraphs | 164 sentences averaging ~25 words | mix in short sentences; let some paragraphs be two sentences |

**The biggest one:** the draft was written with an AI assistant. The honest fix is two-part:
1. Rewrite each section **in your own words**. Close the PDF, explain the section out loud or on paper, then type
   what you said, keeping the numbers and citations.
2. Add the IEEE disclosure to the acknowledgments. Suggested wording: *"The authors used Claude (Anthropic) to
   assist with code development and with drafting and editing parts of the text; all content was reviewed,
   rewritten and verified by the authors, who take full responsibility for it."*

**What genuinely makes writing read as yours:** the specific things only you know.
- why a design choice was made;
- what went wrong (O-SELF never firing, the GSM8K diversity problem, the stalled downloads);
- what surprised you.

AI text is generic. Your project history is not.
