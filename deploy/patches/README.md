# `llm-eval.patch` — attribution & modification notice

`llm-eval.patch` modifies **GROBID** source files (`build.gradle` and
`grobid-trainer/.../evaluation/EndToEndEvaluation.java`). GROBID is
Copyright 2008-present The GROBID contributors and is licensed under the
**Apache License, Version 2.0** — a copy is in [`LICENSE.grobid`](./LICENSE.grobid),
also at <http://www.apache.org/licenses/LICENSE-2.0>.

Per Apache-2.0 §4(b), this file states the changes made to those GROBID files.
The patched portions remain under the Apache License 2.0; only the surrounding
harness (see the repository's `LICENSE`) is MIT.

## What the patch changes

- Adds a `jatsEvalLLM` Gradle task that runs GROBID's own
  `EndToEndEvaluation` over externally produced `*.fulltext.llm[.tag].tei.xml`
  files (a `-Pllmsuffix` selects a specific backend), so an LLM's output is
  scored with GROBID's exact evaluation logic against the JATS/NLM gold. It does
  not run GROBID extraction itself.
- Adds an `LLM` run type to `EndToEndEvaluation` (alongside the existing
  `GROBID`/`PDFX`/`CERMINE`) that reads the LLM TEI suffix instead of
  `*.fulltext.tei.xml`.

The patch is applied at image build time against a pinned GROBID commit (see
`deploy/Dockerfile.harness`); GROBID's source is not vendored in this repository.
