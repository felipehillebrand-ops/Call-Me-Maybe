"""Entry point for the Call Me Maybe function calling tool."""

import sys
from src.cli import parse_args
from src.io_utils import load_functions, load_prompts
from llm_sdk import Small_LLM_Model


def main() -> int:
    """
    Main orchestration function.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    try:
        # TODO: 1. io_utils para ler args.functions_definition e args.input
        # TODO: 2. schemas para validar os JSONs lidos com Pydantic
        # TODO: 3. Carregar o Small_LLM_Model do llm_sdk
        # TODO: 4. Instanciar o generator e iterar sobre os prompts
        # TODO: 5. io_utils para escrever os resultados em args.output

        # Apenas para testar se os argumentos estão sendo lidos corretamente

        print("📂 Reading input files...")
        functions_def = load_functions(args.functions_definition)
        test_prompts = load_prompts(args.input)

        print(f"✅ Found {len(functions_def)} functions "
              f"and {len(test_prompts)} prompts.")

        print("🧠 Loading the LLM (Qwen/Qwen3-0.6B)...")
        llm = Small_LLM_Model()

        print("✅ Model loaded successfully!")

    except FileNotFoundError as e:
        print(f"Error: Missing required file. {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
