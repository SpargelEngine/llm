# Text Pass

Text Pass is a JSON-configurable text processing pipeline. You declare a sequence of **passes** — discrete processing steps — and text streams through them, each pass yielding zero or more output strings.

## Core Concepts

### Model / Instance separation

Every pass has two layers:

- **Model** (Pydantic model): pure configuration — parameters, serializable to JSON
- **Instance** (runtime): created by `model.build(config_path)`, holding compiled regexes, resolved paths, etc.

Passes are available in two forms: **code-constructed** (import classes directly) and **JSON-configured** (via discriminated union with `name` field). Code-constructed passes use base `TextPass` subclasses (`ForEachPass`, `CombinePass`), while JSON-configured passes use `TextPassModel` subclasses (`ForEachPassModel`, `CombinePassModel`) which add Pydantic validation and a `name` discriminator.

### Processing flow

```text
input text stream → [Pass 1] → [Pass 2] → ... → [Pass N] → output text stream
```

Each pass is a generator: `Iterable[str]` → `Iterator[str]`. Some passes change the cardinality (`split_lines` 1→many, `join` many→1, `filter` may drop texts).

### Stream cardinality

| Pass          | Input → Output | Notes                            |
| ------------- | -------------- | -------------------------------- |
| `text`        | N → N+M        | Appends M literal strings        |
| `strip`       | 1 → 1          | One output per input             |
| `split_lines` | 1 → N          | One output per line              |
| `split`       | 1 → N          | One output per segment           |
| `join`        | N → 1          | All inputs merged to one         |
| `replace`     | 1 → 1          | One output per input             |
| `filter`      | 1 → 0 or 1     | Drops non-matching texts         |
| `find`        | N → N+M        | Appends M found file paths       |
| `read_file`   | 1 → 1 or 0     | Skips on decode error (0 output) |
| `for_each`    | 1 → N          | Sub-pipeline cardinality varies  |
| `combine`     | N → M          | M = sum of each pass's outputs   |
| `ref`         | depends        | Depends on referenced passes     |

### Path resolution

All relative paths (`base` fields) resolve relative to the **directory of the config file** (or its parent, if the config path points to a file).

When using `process_texts()` directly, the `path` parameter (default `"."`) is treated as the config path. Pass `path=config_file_path` so that relative bases resolve correctly.

### Common field: `description`

Every pass accepts an optional `description` field for documentation:

```json
{
  "name": "strip",
  "description": "Remove leading/trailing whitespace from each line"
}
```

---

## Quick Start

```python
from spargel_llm.text_pass import TextPassList, process_texts

passes = TextPassList.model_validate({
    "passes": [
        {"name": "strip"},
        {"name": "filter", "pattern": r"keep"},
    ]
}).passes

result = list(process_texts(["  keep me  ", "  drop  "], passes))
# result == ["keep me"]
```

Load and run from a JSON file directly:

```python
from spargel_llm.text_pass import load_texts

for text in load_texts("pipeline.json"):
    print(text)
```

`load_texts` starts with an empty text stream — use passes like `text` or `find` to provide initial content.

---

## Available Passes

### `text` — Provide texts

Append literal strings to the end of the text stream. Input texts pass through unchanged.

| Param | Type        | Description       |
| ----- | ----------- | ----------------- |
| texts | `list[str]` | Strings to append |

```json
{ "name": "text", "texts": ["hello", "world"] }
```

Input `["a"]` → output `["a", "hello", "world"]`.

### `strip` — Strip whitespace

Remove leading and trailing characters from each text. One output per input.

| Param | Type          | Default | Description                                 |
| ----- | ------------- | ------- | ------------------------------------------- |
| chars | `str \| null` | `null`  | Character set to strip; `null` = whitespace |
| right | `bool`        | `false` | If true, only strip trailing (`rstrip`)     |

```json
{ "name": "strip" }
{ "name": "strip", "chars": ".,!", "right": true }
```

### `split_lines` — Split into lines

Split each text by newlines. One text in, zero or more lines out.

| Param     | Type   | Default | Description                  |
| --------- | ------ | ------- | ---------------------------- |
| keep_ends | `bool` | `false` | Whether to keep `\n` endings |

```json
{ "name": "split_lines" }
{ "name": "split_lines", "keep_ends": true }
```

An empty string produces no output (Python's `str.splitlines()` returns `[]` for `""`).

### `split` — Split by separator

Split each text by a literal separator or regex pattern. One text in, zero or more segments out.

| Param     | Type     | Default     | Description                                                                                              |
| --------- | -------- | ----------- | -------------------------------------------------------------------------------------------------------- |
| separator | `str`    | required    | Separator string or regex pattern to split on                                                            |
| regex     | `bool`   | `false`     | Treat `separator` as a regex pattern                                                                     |
| max_split | `int`    | `0`         | Maximum number of splits; `0` means unlimited                                                            |
| behavior  | `string` | `"removed"` | What to do with the separator: `"removed"`, `"isolated"`, `"merged_with_previous"`, `"merged_with_next"` |

```json
{ "name": "split", "separator": "," }
{ "name": "split", "regex": true, "separator": "\\s+" }
{ "name": "split", "separator": "\\t", "max_split": 1 }
{ "name": "split", "regex": true, "separator": "\\n = (?!=).*? = \\n", "behavior": "merged_with_next" }
```

When `regex` is false, uses Python's `str.split` for fast literal splitting (for `"removed"` behavior) or `regex.split` with an escaped literal (for other behaviors). When `regex` is true, uses `regex.split`. An empty string input produces a single empty string output (`"".split(",")` → `[""]`), unlike `split_lines`.

#### Behavior modes

The `behavior` field controls what happens to the matched separator:

| Behavior                 | Input `a,b,c` split by `,` | How it works                               |
| ------------------------ | -------------------------- | ------------------------------------------ |
| `"removed"`              | `["a","b","c"]`            | Separator discarded (default)              |
| `"isolated"`             | `["a",",","b",",","c"]`    | Separator becomes its own segment          |
| `"merged_with_previous"` | `["a,","b,","c"]`          | Separator appended to the preceding piece  |
| `"merged_with_next"`     | `["a",",b",",c"]`          | Separator prepended to the following piece |

Common use case: splitting articles by headings while preserving the heading text. With `behavior: "merged_with_next"`, each section starts with its heading line rather than losing it.

### `join` — Join texts

Concatenate all input texts into a single output string with a separator. Many in, one out.

| Param     | Type  | Default | Description |
| --------- | ----- | ------- | ----------- |
| separator | `str` | `""`    | Separator   |

```json
{ "name": "join", "separator": "\n" }
```

Input `["a", "b", "c"]` → output `["a\nb\nc"]`.

### `replace` — Replace text

Plain or regex substitution, optionally repeating until the text stabilizes. One output per input.

| Param      | Type   | Default  | Description                              |
| ---------- | ------ | -------- | ---------------------------------------- |
| old        | `str`  | required | Substring or regex pattern to replace    |
| new        | `str`  | required | Replacement string                       |
| regex      | `bool` | `false`  | Treat `old` as a regex pattern           |
| repeat     | `bool` | `false`  | Repeat until the result stops changing   |
| max_repeat | `int`  | `1000`   | Maximum iterations when `repeat` is true |

```json
{ "name": "replace", "old": "foo", "new": "bar" }
{ "name": "replace", "regex": true, "old": "\\d+", "new": "NUM" }
{ "name": "replace", "old": "  ", "new": " ", "repeat": true, "max_repeat": 100 }
```

When `repeat` is true, the replacement loop runs until the text stops changing or `max_repeat` iterations are reached — whichever comes first.

### `filter` — Filter texts

Keep or discard texts matching a regex pattern. Each input yields zero or one output.

| Param   | Type   | Default  | Description                         |
| ------- | ------ | -------- | ----------------------------------- |
| pattern | `str`  | required | Regex pattern to match via `search` |
| invert  | `bool` | `false`  | If true, discard matches instead    |

```json
{ "name": "filter", "pattern": "error|warn" }
{ "name": "filter", "pattern": "^#", "invert": true }
```

### `combine` — Independent parallel passes

Apply multiple passes **independently** to the same input stream and concatenate their outputs. Unlike chaining (where each pass feeds into the next), each pass in `combine` sees the full upstream stream. Results are interleaved round-robin.

| Param  | Type                          | Description          |
| ------ | ----------------------------- | -------------------- |
| passes | `list[DiscriminatedTextPass]` | Passes to run independently |

```json
{
  "name": "combine",
  "passes": [
    { "name": "strip" },
    { "name": "split_lines" }
  ]
}
```

Input `["  a\nb  ", "  c  "]` → output (round-robin): `"a\nb"`, `"  a"`, `"c"`, `"b  "`, `"  c  "`.

The strip pass yields `["a\nb", "c"]` while split_lines yields `["  a", "b  ", "  c  "]`. Order among the passes is not significant — use downstream passes to reorder if needed.

**Empty passes**: `"combine"` with an empty `passes` list yields nothing (it returns immediately without consuming the input).

In code, use `CombinePass` directly:

```python
from spargel_llm.text_pass import CombinePass, StripPass, SplitLinesPass

passes = [CombinePass(passes=[StripPass(name="strip"), SplitLinesPass(name="split_lines")])]
```

### `find` — Find files

Walk directories recursively and yield matching file paths. Found paths are **appended** to the existing text stream — input texts pass through unchanged.

| Param        | Type          | Default | Description                                                  |
| ------------ | ------------- | ------- | ------------------------------------------------------------ |
| base         | `str`         | `"."`   | Root search directory                                        |
| paths        | `list[str]`   | `["."]` | Subdirectory paths to search (relative to `base`)            |
| file_pattern | `str \| null` | none    | Filename regex (`fullmatch`)                                 |
| dir_pattern  | `str \| null` | none    | Directory name regex (`fullmatch`); prunes non-matching dirs |

```json
{
  "name": "find",
  "base": "./data",
  "paths": ["corpus", "archive"],
  "file_pattern": ".*\\.txt",
  "dir_pattern": "(sub|top)"
}
```

`file_pattern` matches against filenames, `dir_pattern` matches against directory names to decide which subtrees to enter. Both use `fullmatch` (the entire name must match).

### `read_file` — Read file contents

Read the file at each input path and yield its content as a single text.

| Param       | Type             | Default | Description                            |
| ----------- | ---------------- | ------- | -------------------------------------- |
| base        | `str`            | `"."`   | Base directory for relative paths      |
| encoding    | `str \| null`    | `null`  | File encoding; `null` = system default |
| compression | `"gzip" \| null` | `null`  | Decompression; supports `gzip`         |

```json
{ "name": "read_file", "base": "./data", "encoding": "utf-8" }
{ "name": "read_file", "compression": "gzip" }
```

Each input path is resolved as `base / path`. If the input path is absolute, `base` is ignored (standard `Path` division semantics).

**Error handling:** `UnicodeDecodeError` is logged and the file is skipped (no output for that input). `FileNotFoundError` propagates — wrap in a try/except if you need to handle missing files.

### `for_each` — Per-text sub-pipeline

Apply a sub-pipeline to each input text **in isolation**. The sub-pipeline sees each input as a single-element stream, and all its outputs are collected into the outer stream. Different input texts never mix.

| Param  | Type                          | Description         |
| ------ | ----------------------------- | ------------------- |
| passes | `list[DiscriminatedTextPass]` | Sub-passes to apply |

Typical pattern: strip each line of each text while preserving text boundaries.

```json
{
  "name": "for_each",
  "passes": [
    { "name": "split_lines" },
    { "name": "strip" },
    { "name": "join", "separator": "\n" }
  ]
}
```

Input `["  a\n  b  ", "  c\n  d  "]` → output `["a\nb", "c\nd"]`.

Without `for_each`, the top-level pipeline would interleave lines from different texts. Use `for_each` whenever you need per-text semantics — stripping lines per file, filtering lines within each document, etc.

### `ref` — Reference external passes

Import pass definitions from external JSON files, enabling modular and reusable pipelines.

| Param | Type        | Default  | Description                  |
| ----- | ----------- | -------- | ---------------------------- |
| base  | `str`       | `"."`    | Base directory for ref files |
| paths | `list[str]` | required | JSON file paths to reference |

Referenced files use the same `TextPassList` format as the top-level config. Their `base` fields resolve relative to the referenced file's own directory.

```json
{ "name": "ref", "base": "./configs", "paths": ["clean.json", "filter.json"] }
```

Referenced files can themselves contain `ref` passes (nested references). Each level resolves paths relative to its own file location.

---

## Full Examples

### Simple pipeline (no filesystem)

Strip whitespace, discard empty lines, join with newlines:

```json
{
  "passes": [
    { "name": "split_lines" },
    { "name": "strip" },
    { "name": "filter", "pattern": "^$", "invert": true },
    { "name": "join", "separator": "\n" }
  ]
}
```

Input `"  hello  \n\n  world  "` → output `"hello\nworld"`.

### Batch-clean text files

```json
{
  "passes": [
    {
      "name": "find",
      "base": "./corpus",
      "paths": ["."],
      "file_pattern": ".*\\.txt"
    },
    {
      "name": "read_file",
      "base": "./corpus",
      "encoding": "utf-8"
    },
    {
      "name": "for_each",
      "passes": [
        { "name": "split_lines" },
        { "name": "strip" },
        { "name": "filter", "pattern": "^$", "invert": true },
        { "name": "join", "separator": "\n" }
      ]
    }
  ]
}
```

What this does:

1. `find` — discover all `.txt` files under `./corpus`
2. `read_file` — read each file's content
3. `for_each` — for each file:
   - split into lines
   - strip whitespace from each line
   - drop empty lines
   - rejoin with newlines

### Modular pipeline with `ref`

Main config `pipeline.json`:

```json
{
  "passes": [
    { "name": "ref", "paths": ["find-read.json"] },
    { "name": "ref", "paths": ["clean.json"] }
  ]
}
```

`find-read.json`:

```json
{
  "passes": [
    {
      "name": "find",
      "base": "./corpus",
      "paths": ["."],
      "file_pattern": ".*\\.txt"
    },
    { "name": "read_file", "base": "./corpus", "encoding": "utf-8" }
  ]
}
```

`clean.json`:

```json
{
  "passes": [
    {
      "name": "for_each",
      "passes": [
        { "name": "split_lines" },
        { "name": "strip" },
        { "name": "join", "separator": "\n" }
      ]
    }
  ]
}
```

---

## Architecture

### Type system

All JSON-configurable pass types form a discriminated union via `DiscriminatedTextPass`, using Pydantic's `Discriminator("name")` to dispatch to the correct Model by the `name` field. The type alias is `TextPassModelUnion`.

Each pass that supports JSON configuration uses a **Model** suffix: `ForEachPassModel`, `CombinePassModel`. The base versions (`ForEachPass`, `CombinePass`) are plain code classes accepting `list[TextPass]` — they work without Pydantic validation and are used when constructing pipelines directly in Python.

### Internal components

| Component              | Role                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| `TextPass`             | Base factory interface; defines `build(config_path) → TextPassInstance` (code-constructed passes)           |
| `TextPassModel`        | Base for JSON-configurable passes; extends `TextPass` + `BaseModel`; defines `description` field              |
| `TextPassInstance`     | Base for all pass runtimes; defines `process(texts) → Iterator[str]`                                         |
| `ChainPassInstance`    | Chains multiple instances, executing them in sequence                                                        |
| `ForEachPassInstance`  | Applies a sub-pipeline to each input text in isolation                                                       |
| `CombinePassInstance`  | Runs multiple instances independently on the same input, round-robin interleave                              |
| `TextPassList`         | Top-level model wrapping `passes: list[DiscriminatedTextPass]`                                               |
| `process_texts()`      | Public API: build instances from passes, chain, and run. Signature: `process_texts(texts, passes, path=".")` |
| `load_texts()`         | Convenience: load config from a JSON file and run with empty initial stream                                  |

### Generate JSON Schema

```bash
uv run python scripts/text_pass_schema.py
```

Outputs the full JSON Schema for `TextPassList`, usable for editor autocompletion and validation.
