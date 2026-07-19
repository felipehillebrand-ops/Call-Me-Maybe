UV			= uv
SRC			= src
CALLING		= data/input/function_calling_tests.json
DEFINITION	= data/input/functions_definition.json
OUTPUT		= data/output/function_calling_results.json

HF_CACHE := $(shell if [ -d /sgoinfre/$(USER) ]; then \
	echo /sgoinfre/$(USER)/.cache/huggingface; \
	else echo $$HOME/.cache/huggingface; fi)

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
	@echo ">>> Hugging Face cache: $(HF_CACHE)"
	@mkdir -p $(HF_CACHE)
	HF_HOME="$(HF_CACHE)" $(UV) run python -m $(SRC) \
		--functions_definition $(DEFINITION) \
		--input $(CALLING) \
		--output $(OUTPUT)

run-verbose:
	@echo ">>> Running function calling tool (verbose generation trace)..."
	@echo ">>> Hugging Face cache: $(HF_CACHE)"
	@mkdir -p $(HF_CACHE)
	HF_HOME="$(HF_CACHE)" $(UV) run python -m $(SRC) \
		--functions_definition $(DEFINITION) \
		--input $(CALLING) \
		--output $(OUTPUT) \
		--verbose
 
run-trace:
	@echo ">>> Running function calling tool (saving generation trace to JSON)..."
	@echo ">>> Hugging Face cache: $(HF_CACHE)"
	@mkdir -p $(HF_CACHE)
	HF_HOME="$(HF_CACHE)" $(UV) run python -m $(SRC) \
		--functions_definition $(DEFINITION) \
		--input $(CALLING) \
		--output $(OUTPUT) \
		--trace-output data/output/trace.json

INDEXES ?= 1,6

error:
	@echo ">>> Running function calling tool with simulated failures on"
	@echo ">>> prompt(s) #$(INDEXES) (CALL_ME_MAYBE_DEMO_FAIL_INDEXES)..."
	@echo ">>> Hugging Face cache: $(HF_CACHE)"
	@mkdir -p $(HF_CACHE)
	CALL_ME_MAYBE_DEMO_FAIL_INDEXES="$(INDEXES)" \
	HF_HOME="$(HF_CACHE)" $(UV) run python -m $(SRC) \
		--functions_definition $(DEFINITION) \
		--input $(CALLING) \
		--output $(OUTPUT) \
		--verbose

debug:
	@echo ">>> Running function calling tool in debug mode..."
	@echo ">>> Hugging Face cache: $(HF_CACHE)"
	@mkdir -p $(HF_CACHE)
	HF_HOME="$(HF_CACHE)" $(UV) run python -m pdb -m $(SRC) \
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
	@echo "    run-verbose  Run the tool printing a live generation trace (--verbose)"
	@echo "    run-trace    Run the tool saving the generation trace to data/output/trace.json"
	@echo "    error        Run the tool forcing simulated failures (default: prompts #1,#6;"
	@echo "                 override with 'make error INDEXES=3,7') to demo error recovery"
	@echo "    debug        Run the tool with Python's pdb debugger"
	@echo "    clean        Remove __pycache__, .mypy_cache, .pytest_cache, *.pyc, *.pyo"
	@echo "    fclean       clean + remove the uv virtual environment"
	@echo "    lint         Run flake8 + mypy with mandatory flags"
	@echo "    lint-strict  Run flake8 + mypy with --strict"
	@echo ""

.PHONY: all install run run-verbose run-trace error debug clean fclean lint lint-strict help
