"""Entry point for the Call Me Maybe function calling tool."""

import sys
from src.cli import parse_args
from src.io_utils import load_functions, load_prompts, save_results
from src.generator import ConstrainedGenerator
from llm_sdk import Small_LLM_Model  # type: ignore


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

        print("⚙️ Initializing constrained decoding pipeline...")
        generator = ConstrainedGenerator(llm)

        print("🚀 Executing function calling on prompts...")
        results = []
        for idx, test_prompt in enumerate(test_prompts, 1):
            print(f"  [{idx}/{len(test_prompts)}] "
                  f"Processing: '{test_prompt.prompt}'")
            output = generator.generate_call(test_prompt, functions_def)
            results.append(output)

        print("💾 Saving structured results...")
        save_results(results, args.output)
        print(f"🎉 Process complete! Results saved to: {args.output}")

    except FileNotFoundError as e:
        print(f"Error: Missing required file. {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
