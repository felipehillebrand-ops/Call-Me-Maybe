"""Entry point for the Call Me Maybe function calling tool."""

import sys
import traceback
from typing import Any, Dict, List
from src.cli import parse_args
from src.io_utils import load_functions, load_prompts, save_results
from src.generator import ConstrainedGenerator
from src.schemas import FunctionCallOutput, FunctionDefinition
from llm_sdk import Small_LLM_Model


_DEFAULT_BY_TYPE: Dict[str, Any] = {
    "string": "",
    "number": 0.0,
    "integer": 0,
    "boolean": False,
}


def _build_failure_placeholder(
    prompt: str, functions_def: List[FunctionDefinition]
) -> FunctionCallOutput:
    """
    Build a schema-compliant placeholder for a prompt that failed to
    generate a function call.

    Args:
        prompt (str): The original natural-language prompt that failed.
        functions_def (List[FunctionDefinition]): Available function
            definitions, used to pick a fallback function and its
            default argument values.

    Returns:
        FunctionCallOutput: A placeholder result matching the schema of
        the fallback function.
    """
    if not functions_def:
        return FunctionCallOutput(prompt=prompt, name="", parameters={})

    fallback_fn = functions_def[0]
    default_params = {
        p_name: _DEFAULT_BY_TYPE.get(p_def.type, "")
        for p_name, p_def in fallback_fn.parameters.items()
    }
    return FunctionCallOutput(
        prompt=prompt, name=fallback_fn.name, parameters=default_params
    )


def _load_model(model_name: str) -> Small_LLM_Model:
    """
    Helper to instantiate the LLM model dynamically.

    Tries keyword and positional model initialization to ensure compatibility
    across different SDK wrapper signatures.
    """
    try:
        return Small_LLM_Model(model_name=model_name)
    except TypeError:
        try:
            return Small_LLM_Model(model_name)
        except TypeError:
            return Small_LLM_Model()


def main() -> int:
    """
    Main orchestration function.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    try:
        print("📂 Reading input files...")
        functions_def = load_functions(args.functions_definition)
        test_prompts = load_prompts(args.input)
        print(f"✅ Found {len(functions_def)} functions "
              f"and {len(test_prompts)} prompts.")

        print(f"🧠 Loading the LLM ({args.model})...")
        llm = _load_model(args.model)
        print("✅ Model loaded successfully!")

        print("⚙️  Initializing constrained decoding pipeline...")
        generator = ConstrainedGenerator(llm, verbose=args.verbose)

        print("🚀 Executing function calling on prompts...")
        results: List[FunctionCallOutput] = []
        failures: List[Dict[str, Any]] = []
        success_count = 0

        for idx, test_prompt in enumerate(test_prompts, 1):
            print(f"  [{idx}/{len(test_prompts)}] "
                  f"Processing: '{test_prompt.prompt}'")
            try:
                output = generator.generate_call(test_prompt, functions_def)
                results.append(output)
                success_count += 1
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                if tb:
                    failed_in = tb[-1].name
                else:
                    failed_in = "unknown"
                print(
                    f"    ⚠️  Failed in '{failed_in}' "
                    f"({type(e).__name__}: {e}); recording a placeholder "
                    f"to keep prompt-to-result ordering intact.",
                    file=sys.stderr,
                )
                failures.append({
                    "prompt": test_prompt.prompt,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "failed_in": failed_in,
                })
                results.append(_build_failure_placeholder(
                    test_prompt.prompt, functions_def
                ))

        print("💾 Saving structured results...")
        save_results(results, args.output)
        print(f"🎉 Process complete! {success_count}/{len(test_prompts)} "
              f"prompt(s) succeeded. Results saved to: {args.output}")

        if failures:
            print(
                f"⚠️  {len(failures)} prompt(s) failed and were recorded "
                f"as schema-compliant fallback placeholders (prompt kept, "
                f"fallback function name with type-correct default "
                f"parameters):",
                file=sys.stderr,
            )
            for failure in failures:
                print(
                    f"    - {failure['prompt']!r}: "
                    f"{failure['error_type']} in "
                    f"'{failure['failed_in']}': "
                    f"{failure['error_message']}",
                    file=sys.stderr,
                )

        if args.trace_output:
            generator.export_trace(args.trace_output)
            print(f"🔎 Generation trace saved to: {args.trace_output}")

    except FileNotFoundError as e:
        print(f"Error: Missing required file. {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"Error: Invalid input data. {e}", file=sys.stderr)
        return 1

    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            failed_in = tb[-1].name
        else:
            failed_in = "unknown"
        print(
            f"Error: An unexpected error occurred in '{failed_in}': "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
