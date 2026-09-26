# Collection

`data/sample/` is a four-topic worked example. Its wording is illustrative and
was **not** verified against the official pages, so it is used only by the
tests and the CI smoke run. Nothing from it may appear in a reported result.

The real collection goes in `data/v1/` and is built from official sources. Its
collection protocol is added with the formal data in a later reviewed branch.

## Files

| File | One row per | Columns |
|---|---|---|
| `sources.csv` | official page | `source_id, organisation, title, url, access_date, risk_category` |
| `passages.csv` | passage | `passage_id, topic_id, source_id, section_heading, text` |
| `topics.csv` | information need | `topic_id, category, knowledge_type, information_need` |
| `questions.csv` | question | `question_id, topic_id, question_form, language, text` |
| `qrels.txt` | judgement | `question_id 0 passage_id grade` |
| `splits.csv` | topic | `topic_id, split` |

## Rules the validator enforces

* `risk_category` is one of general, visa, health, employment, housing, emergency.
* A passage is 40–300 words and has a section heading.
* `knowledge_type` is `known` (the answer is stated in one passage) or
  `inferred` (the answer has to be put together from more than one).
* `question_form` is `canonical`, `paraphrased` or `ookb`, and every question
  uses the `en` language code.
* Every answerable question has at least one judgement; every `ookb` question
  has none and belongs to no topic.
* Grades are 1 (partially relevant) or 2 (fully answers). Graded judgements are
  what make NDCG meaningful — a binary hit rate would treat both the same.
* Splits are assigned per **topic**, so a topic's canonical and paraphrased
  questions always land on the same side of the validation/test boundary.
  Anything else leaks the test set.

Run `mm --data data/v1 validate` after every editing session. It is quick and
it fails loudly.
