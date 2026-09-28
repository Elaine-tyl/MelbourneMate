# Blind answer review

Score each answer without opening `key.csv`.

- `correctness`: 0 wrong, 1 partly correct, 2 correct.
- `evidence_support`: 0 unsupported, 1 partly supported, 2 fully supported.
- `fallback_appropriateness`: 0 wrong action, 1 acceptable but weak, 2 appropriate.

Use 0 for evidence support when no retrieved evidence supports an answer. For an
out-of-knowledge-base question, score fallback appropriateness against the need
to refuse safely. Add a short note only when a score needs explanation.
