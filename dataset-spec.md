# Dataset spec v0.1 (Week 3)

> Defaults below are chosen by the scripts. For each "DECISION" line, write ONE sentence of
> reasoning in your own words - this becomes your thesis chapter 3 (dataset engineering).

## 1. Unit of data
One example = (instruction, data) pair in the Frege format:

    [FUNCTION_START] <instruction> [FUNCTION_END] [ARGUMENT_START] <data> [ARGUMENT_END]

function = trusted instruction, argument = untrusted data it is applied to.

## 2. Labeling rule (most important definition)
**label = 1 (attack)** iff the data contains text that tries to make the assistant deviate from
its given instruction: change the task, take on a role, leak the prompt/data, produce
attacker-chosen output or links.
**label = 0** otherwise - including harmless imperatives aimed at a *human reader* of the document
("Ignore the footnotes below.") and passages that merely mention words like "ignore" or "password".

This is the operational version of `Instr(x)` from the Frege/quantifier slide:
Attack(d) = there exists a span x in d with Instr(x).
DECISION: why this rule? (what about borderline cases? who decides?)  -> _write here_

## 3. Variants (columns: `variant`)
| variant | what it is | label |
|---|---|---|
| benign | squad_v2 passage + one task instruction | 0 |
| attack | the SAME passage + instruction with one injected sentence (matched twin) | 1 |
| hard_negative | `benign_imperative` (harmless note inserted) or `keyword_natural` (real passage with suspicious words) | 0 |

Matched twins differ only by the injection, so a model cannot win by learning document style.
Twins and hard negatives of one passage share a `doc_id`.

## 4. Attack families (column: `attack_family`)
ignore_previous, role_hijack, task_switch, prompt_leak, exfiltration (markdown-image URL),
phishing, fake_markers (forged Frege markers), deepset_real (real prompts, English-filtered).
NOT yet included: HouYi-style separator attacks (Week 4) -> will become a second held-out family.
DECISION: why these families? which papers justify them? -> _write here_

## 5. Controls against shortcuts
- Injection **position** randomised (start / middle / end).
- The same **style prefixes** ("Note:", "P.S.", ...) are used for attacks and harmless notes.
- Non-English deepset payloads are dropped (heuristic) so language is not a shortcut.
- Instructions are identical within a twin pair -> an instruction-only model must score ~chance (sanity check in script 09).

## 6. Marker escaping
Reserved markers in instruction or data are rewritten `[X_Y]` -> `(X_Y)` (case/space variants and nested
forgeries included; see `tests/test_frege_format.py`). LIMITATION: string-level only. A real system
needs reserved tokens that user text can never produce (StruQ-style). State this in the thesis.

## 7. Splits
- `test_ood`: every document whose attack twin uses the held-out family (default `phishing`).
- Remaining documents split by `doc_id` 80/10/10 -> train / val / test.
- Checks: no shared `doc_id`, `text` or `data` between splits; held-out family absent from train/val/test.

## 8. Known limitations (copy into the thesis limitations section)
- Passages come from one source (Wikipedia via squad_v2); real RAG data (emails, web pages) differs.
- Attacks are synthetic/templated; real attackers adapt. Held-out families only partly test this.
- Payloads are short single sentences; multi-step or obfuscated attacks are not covered yet.
- Label rule is a design choice; borderline cases exist.
