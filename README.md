*This project has been created as part of the 42 curriculum by fjose-hi.*

# Call Me Maybe — Introduction to function calling in LLMs

## Description

**call me maybe** is a function calling tool that translates natural language
prompts into structured, machine-executable function calls. Given a question
like *"What is the sum of 40 and 2?"*, the program does not answer *"42"* —
instead it outputs the function to call and its typed arguments:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The goal of the project is to make a small, unreliable language model
(**Qwen/Qwen3-0.6B**, ~500M parameters) produce **100% valid, schema-compliant
JSON**, without ever relying on the model "getting lucky" with prompting
alone. This is achieved through **constrained decoding**: at every generation
step, the model's logits are masked so that only tokens compatible with both
valid JSON syntax and the expected function schema can be selected.

The program reads two input files — a list of available functions
(`functions_definition.json`) and a list of natural language prompts
(`function_calling_tests.json`) — and produces a single output file
(`function_calling_results.json`) containing, for every prompt, the selected
function name and its extracted arguments.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package, expected to sit alongside `src/` (already included
  in this repository)

### Installation

```sh
make install
```

This installs `uv` if missing and runs `uv sync` to create the virtual
environment and install all dependencies (`numpy`, `pydantic`, `llm-sdk`,
`accelerate`).

### Running the program

```sh
make run
```

which is equivalent to:

```sh
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Other Makefile targets

| Target        | Description                                              |
|---------------|-----------------------------------------------------------|
| `install`     | Install dependencies via `uv sync`                        |
| `run`         | Run the tool with the default input/output paths          |
| `run-verbose` | Run the tool printing a live generation trace (`--verbose`) |
| `run-trace`   | Run the tool saving the generation trace to `data/output/trace.json` |
| `error`       | Run the tool forcing simulated failures on prompts #1 and #6 (override with `make error INDEXES=3,7`), to demo error recovery |
| `debug`       | Run the tool under Python's `pdb` debugger                 |
| `clean`       | Remove `__pycache__`, `.mypy_cache`, `.pytest_cache`, etc. |
| `fclean`      | `clean` + remove the `uv` virtual environment           |
| `lint`        | Run `flake8` and `mypy` with the mandatory flags            |
| `lint-strict` | Run `flake8` and `mypy --strict`                            |

## Algorithm Explanation

The core idea of constrained decoding is simple: **generate the JSON output
token by token, and at each step forbid every token that would break either
JSON syntax or the function's schema.** Concretely, the model's raw logits
are never sampled directly — they are first masked by setting every
disallowed token's logit to `-inf`, and only then is `argmax` applied
(`ConstrainedGenerator._select_next_token`). This guarantees that whatever
token is chosen, the overall output remains valid.

The pipeline for a single prompt is:

1. **Fixed scaffolding is injected directly as tokens.** The literal parts
   of the JSON structure (`{`, `"prompt": "..."`, `"name": "`, `"parameters":
   {`, commas, closing braces, etc.) are encoded once and appended to
   `input_ids` — the model never has to "choose" to produce them, so they can
   never be malformed.

2. **Function name selection** (`_generate_function_name`). All function
   names are encoded into token sequences. At each step, only tokens that
   extend at least one *still-active* candidate name are allowed. As tokens
   are generated, candidates that no longer match are dropped, until exactly
   one name remains. This makes the model **pick**, rather than **spell**,
   the function name — it is structurally impossible to output a function
   name that doesn't exist.

3. **Parameter generation**, one parameter at a time, dispatched by declared
   type:
   - **`string`** (`_generate_string_parameter`): value tokens are only
     allowed if the text decoded so far remains an exact substring of the
     original prompt, anchored at some starting offset. This forces the
     model to *extract* text that actually appears in the prompt instead of
     hallucinating new content. A repetition penalty, an increasing "stop
     bias" toward the closing quote, and n-gram degeneration detection keep
     the value from looping or running on indefinitely. Trailing words that
     just echo the parameter's own name (e.g. avoid `"hello string"` for a
     parameter literally called `string`) are trimmed at the end.
   - **`number` / `integer`** (`_generate_number_parameter`): only digits,
     `.`, `-`, and the JSON delimiters that may legally follow a number
     (`,` or `}`) are allowed at each step. The accumulated digits are then
     cast to `float` (if the type is `number` or a `.` was produced) or
     `int` otherwise.
   - **`boolean`** (`_generate_boolean_parameter`): the same prefix-narrowing
     strategy is applied over the two JSON literals `true` and `false`.

4. The finished structure is parsed into a `FunctionCallOutput` (Pydantic
   model) and appended to the results list, which is serialized to
   `function_calling_results.json` at the end.

Because every branch of generation is constrained at the token level, the
output is **syntactically guaranteed to be valid JSON matching the schema**
— the model's only real freedom is *which* valid token to pick, not whether
the result parses.

## Design Decisions

- **Pydantic everywhere.** All data crossing a boundary (input files, LLM
  outputs, on-disk results) is validated through Pydantic models
  (`schemas.py`), so malformed input files or unexpected structures are
  caught early with clear errors instead of causing silent corruption
  further down the pipeline.
- **Vocabulary pre-filtering (`VocabFilter`).** Classifying every token in
  the vocabulary once, up front, into categories (numeric, quote-containing,
  etc.) avoids repeatedly decoding the same token during generation, which
  keeps the per-step constrained decoding loop fast.
- **String extraction is anchored to the prompt, not free generation.**
  Given the model is only 500M parameters, letting it generate arbitrary
  text for string parameters would be unreliable. Anchoring every candidate
  token to a valid continuation of a substring of the prompt turns the task
  into extraction rather than generation, which small models are much
  better at.
- **Only the public `llm_sdk` interface is used**
  (`encode`, `decode`, `get_logits_from_input_ids`) — no private attributes
  or methods of the SDK are accessed, as required by the subject.
- **CLI defaults mirror the subject's expected directory layout**
  (`data/input/...`, `data/output/...`) while still allowing full overrides
  via `--functions_definition`, `--input`, and `--output`.

## Performance Analysis

- **Validity:** every output is constructed token-by-token under an
  explicit grammar/schema mask, so **100% of generated entries are valid,
  parseable JSON matching the expected schema** — this is a structural
  guarantee, not a statistical one.
- **Accuracy:** function selection and string-parameter extraction were
  validated against the provided `function_calling_tests.json` /
  `functions_definition.json` pairs, correctly identifying the intended
  function and its arguments across the different function families (math,
  string manipulation, regex substitution).
- **Speed:** processing the provided test set completes well within the
  5-minute budget on standard (CPU) hardware; most of the per-prompt cost is
  the string-parameter loop, bounded to at most 25 decoding steps per
  string field.
- **Reliability:** since Qwen3-0.6B alone is described in the subject as
  succeeding only ~30% of the time at producing structured JSON, the
  constrained decoding approach is what pushes reliability to effectively
  100% validity, independent of how "confident" the underlying model is.

## Challenges Faced

- **Preventing degenerate loops in free-form string generation.** Left
  unconstrained, small models tend to repeat short n-grams indefinitely.
  This was solved with a combination of a repetition penalty on already-used
  tokens and explicit n-gram repeat detection (`_is_degenerate_repeat`) that
  forces an early stop.
- **Knowing *when* to stop a string value.** Since the model could
  technically keep extending a valid substring of the prompt forever, a
  growing "stop bias" toward the closing quote token was introduced, so the
  longer a value gets, the more the model is nudged to close it once a
  reasonable point is reached.
- **Resolving `int` vs `float` for numeric parameters.** The schema
  distinguishes `"number"` from `"integer"`, but token-level generation only
  produces raw digit sequences. The final string is inspected for a decimal
  point (or the declared type is checked) before casting, ensuring the
  Python type matches what the schema promises.

## Testing Strategy

- The provided `function_calling_tests.json` / `functions_definition.json`
  pair was used as the primary test set, covering arithmetic (`fn_add_numbers`,
  `fn_get_square_root`), string manipulation (`fn_greet`, `fn_reverse_string`),
  and regex substitution (`fn_substitute_string_with_regex`) — exercising
  every declared parameter type.
- Output files were validated by loading them back with `json.load` and
  checking every entry against `FunctionCallOutput` to confirm 100%
  parseability and schema compliance.
- Edge cases exercised manually: prompts with numbers embedded in strings
  (e.g. `"Hello 34 I'm 233 years old"`), prompts requiring exact substring
  extraction with punctuation, and missing/malformed input files (to confirm
  graceful error handling instead of crashes).

## Example Usage

```sh
$ make run
📂 Reading input files...
✅ Found 5 functions and 11 prompts.
🧠 Loading the LLM (Qwen/Qwen3-0.6B)...
✅ Model loaded successfully!
⚙️ Initializing constrained decoding pipeline...
🔍 Pre-filtering vocabulary tokens (this may take a moment)...
✅ Vocabulary filtering complete.
🚀 Executing function calling on prompts...
  [1/11] Processing: 'What is the sum of 2 and 3?'
  ...
💾 Saving structured results...
🎉 Process complete! Results saved to: data/output/function_calling_results.json
```

Resulting `data/output/function_calling_results.json` excerpt:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Bonus Features

The following bonus features from the subject's "Bonus Part" chapter are
implemented and working:

### Visualization of the generation process

Two opt-in CLI flags expose the constrained decoding process step by step,
without changing the program's default behavior or output when unused:

- **`--verbose`** (`make run-verbose`): prints every decoding step live to
  stdout as each prompt is processed — function-name candidate narrowing,
  parameter type dispatch, each token chosen for string/number/boolean
  values, and why generation stopped (delimiter reached, stop token
  selected, no extending candidates left, degenerate repeat detected).
- **`--trace-output <path>`** (`make run-trace`): saves the same
  step-by-step trace as structured JSON (`data/output/trace.json` by
  default), so it can be inspected or visualized after the run instead of
  scrolling through terminal output.

Both are implemented via a single hook, `ConstrainedGenerator._log_step`,
called at every masking/selection decision throughout
`_generate_function_name`, `_generate_string_parameter`,
`_generate_number_parameter`, and `_generate_boolean_parameter`. The trace
is always recorded internally (negligible cost — a dictionary append per
step); only printing to stdout is gated behind `--verbose`, and only
writing to disk is gated behind `--trace-output`. Neither flag is used by
default, so `make run` / `uv run python -m src` alone keeps producing
**only** the single `function_calling_results.json` file required by the
subject.

### Demonstration of how encoding and decoding integrate with constrained decoding

The trace produced by the visualization feature above doubles as a direct,
inspectable demonstration of this integration: every `token_piece` recorded
in the trace comes from `self.llm.decode([token_id])`, called immediately
after a token is chosen by the logit-masking step — showing, token by
token, how `decode` turns the model's selected ID back into the text that
ends up anchored against the prompt or accumulated into a parameter value.
Symmetrically, `self.llm.encode(...)` is used to build every set of
candidate token sequences that the mask is built from (function names in
`_generate_function_name`, and the `"true"`/`"false"` literals in
`_generate_boolean_parameter`). Running `make run-verbose` on a single
prompt makes this encode → mask → select → decode loop directly observable.

### Advanced error recovery mechanisms

Each prompt is now processed in its own isolated `try/except` block inside
`__main__.py`, instead of one `try` wrapping the entire batch. If
`generate_call` raises for one prompt, the failure is logged (prompt text,
exception type, message, and the function where it occurred) to `stderr`,
and the loop **continues** with the remaining prompts rather than aborting
the whole run. At the end, the summary line reports how many prompts
succeeded (e.g. `8/11 prompt(s) succeeded`).

Crucially, a failed prompt still gets **an entry in the output list** — a
placeholder with the original `prompt` and empty `name`/`parameters` —
instead of being skipped outright. This matters because the output is a
JSON *array*, and any position-based consumer (including this project's own
moulinette, which compares `results[i]` against `test_prompts[i]`) relies on
that array staying the same length and order as the input prompts. Silently
dropping a failed slot would shift every *subsequent* result one position
to the left, turning one real failure into a cascade of false failures for
every prompt after it — the opposite of graceful degradation. Keeping the
slot (even if empty) means only the genuinely failed prompt(s) fail
grading; everything before and after stays correctly aligned and unaffected.
This intentionally does **not** write any extra output file, so the
program's default output stays exactly the single JSON file required by
the subject.

To demonstrate this without depending on a naturally-occurring failure, a
demo-only, opt-in trigger is available: setting the environment variable
`CALL_ME_MAYBE_DEMO_FAIL_INDEXES` to a comma-separated list of 1-based
prompt numbers (matching the `Prompt #N` shown in `--verbose` output)
simulates a failure on exactly those prompts, e.g.:

```sh
# Simulates a failure on prompts #1 and #6 only
CALL_ME_MAYBE_DEMO_FAIL_INDEXES="1,6" uv run python -m src --verbose

# Equivalent shortcut via Makefile (same default indexes, --verbose on):
make error

# Override which prompts fail:
make error INDEXES=3,7
```

Matching by index rather than by a text substring is intentional: an
earlier version of this trigger matched by substring (e.g. `"product"`),
which also accidentally matched unrelated prompts containing that
substring (e.g. `"production"`), triggering more failures than intended.
Index-based matching has no such collision risk. The variable is unset by
default, so it has zero effect on normal runs (including the moulinette) —
it exists purely to make the recovery mechanism demonstrable on request
during evaluation.

## Resources

### References

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [Hugging Face — Constrained decoding / Guided generation concepts](https://huggingface.co/docs/transformers/main/en/generation_strategies)
- [Qwen3 model card (Qwen/Qwen3-0.6B)](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)

### Use of AI

AI assistance (Claude by Anthropic and Gemini by Google) was used during this project for:

- Guidance on project structure and file organization.
- Code review for conformance with the project specification.
- Reviewing the implementation (`generator.py`, `schemas.py`, `io_utils.py`,
  `cli.py`, `vocab.py`, `__main__.py`) against the subject to identify gaps
- Review of type hints and mypy compliance.
- Improving docstrings across `generator.py`, `schemas.py`, `io_utils.py`,
  `cli.py`, `vocab.py`, `__main__.py`, and the `llm_sdk.pyi`, to comply
  with the PEP 257 documentation requirement.

All AI-generated or AI-suggested code was read, understood, and verified
(syntax-checked and cross-checked against the subject) before being
included in this repository.
