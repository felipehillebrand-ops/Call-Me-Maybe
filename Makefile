UV			= uv
SRC			= src
CALLING		= data/input/function_calling_tests.json
DEFINITION	= data/input/functions_definition.json
OUTPUT		= data/output/function_calling_results.json
 
all: install run

install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "✅ >>> uv is already installed ($$(uv --version))"; \
	elif [ -f "$$HOME/.local/bin/uv" ]; then \
		echo "✅ >>> uv found in $$HOME/.local/bin"; \
	else \
		echo "📦 >>> uv not found. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "📦 >>> Installing dependencies..."
	@PATH="$$HOME/.local/bin:$$PATH" uv sync
	@echo "✅ >>> Installation complete."
 
run:
	@echo ">>> Running function calling tool..."
	HF_HOME="/sgoinfre/fjose-hi/.cache/huggingface" $(UV) run python -m $(SRC) \
		--functions_definition $(DEFINITION) \
		--input $(CALLING) \
		--output $(OUTPUT)
 
debug:
	@echo ">>> Running function calling tool in debug mode..."
	HF_HOME="/sgoinfre/fjose-hi/.cache/huggingface" $(UV) run python -m pdb -m $(SRC) \
		--functions_definition $(DEFINITION) \
		--input $(CALLING) \
		--output $(OUTPUT)

clean:
	@echo "🧹 >>> Cleaning temporary files..."
	find . -type d -name "__pycache__"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"         -delete 2>/dev/null || true
	find . -type f -name "*.pyo"         -delete 2>/dev/null || true
	@echo "✅ >>> Partial Clean complete."
 
fclean: clean
	@echo "🧹 >>> Removing uv virtual environment..."
	rm -rf .venv
	@echo "✅ >>> Full clean complete."

lint:
	@echo "🔍 >>> Running flake8..."
	$(UV) run flake8 .
	@echo "🧠 >>> Running mypy..."
	$(UV) run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
	@echo "✅ >>> Lint complete."
 
lint-strict:
	@echo "🔍 >>> Running flake8 (strict)..."
	$(UV) run flake8 .
	@echo "🧠 >>> Running mypy (strict)..."
	$(UV) run mypy . --strict
	@echo "✅ >>> Strict lint complete."

help:
	@echo ""
	@echo "  Call Me Maybe — Function Calling in LLMs"
	@echo ""
	@echo "  Targets:"
	@echo "    install      Install project dependencies via uv sync"
	@echo "    run          Run the function calling tool"
	@echo "    debug        Run the tool with Python's pdb debugger"
	@echo "    clean        Remove __pycache__, .mypy_cache, .pytest_cache, *.pyc, *.pyo"
	@echo "    fclean       clean + remove the uv virtual environment"
	@echo "    lint         Run flake8 + mypy with mandatory flags"
	@echo "    lint-strict  Run flake8 + mypy with --strict"
	@echo ""
	@echo "  Example:"
	@echo "    make run"
	@echo "    make run CALLING=data/input/other_tests.json"
	@echo ""

.PHONY: all install run debug clean fclean lint lint-strict help