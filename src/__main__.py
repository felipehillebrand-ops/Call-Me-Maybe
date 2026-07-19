"""Entry point for the Call Me Maybe function calling tool."""

import sys
import traceback
from typing import Any, Dict, List
from src.cli import parse_args
from src.io_utils import load_functions, load_prompts, save_results
from src.generator import ConstrainedGenerator
from src.schemas import FunctionCallOutput
from llm_sdk import Small_LLM_Model


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

        print("🧠 Loading the LLM (Qwen/Qwen3-0.6B)...")
        llm = Small_LLM_Model()
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
                failed_in = tb[-1].name if tb else "unknown"
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
                results.append(FunctionCallOutput(
                    prompt=test_prompt.prompt, name="", parameters={}
                ))

        print("💾 Saving structured results...")
        save_results(results, args.output)
        print(f"🎉 Process complete! {success_count}/{len(test_prompts)} "
              f"prompt(s) succeeded. Results saved to: {args.output}")

        if failures:
            print(
                f"⚠️  {len(failures)} prompt(s) failed and were recorded "
                f"as empty placeholders (prompt kept, name/parameters "
                f"empty):",
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
        failed_in = tb[-1].name if tb else "unknown"
        print(
            f"Error: An unexpected error occurred in '{failed_in}': "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
