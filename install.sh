#!/bin/bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
TORCH_VERSION=${TORCH_VERSION:-2.10.0}
TORCH_CUDA_VERSION=${TORCH_CUDA_VERSION:-cu130}
UV_VLLM_VERSION=${UV_VLLM_VERSION:-0.19.1}
UV_PYTHON_VERSION=${UV_PYTHON_VERSION:-3.12}
UV_INSTALL_METHOD=${UV_INSTALL_METHOD:-standalone}

OS=$(uname -s)

readonly DATASET_LOADER_SCRIPT="src/dataset_loader.py"
readonly DEFAULT_OUTPUT_DIR="data"

if [[ -t 1 ]]; then
    readonly BOLD=$(tput bold 2>/dev/null || echo)
    readonly RED=$(tput setaf 1 2>/dev/null || echo)
    readonly GREEN=$(tput setaf 2 2>/dev/null || echo)
    readonly YELLOW=$(tput setaf 3 2>/dev/null || echo)
    readonly BLUE=$(tput setaf 4 2>/dev/null || echo)
    readonly RESET=$(tput sgr0 2>/dev/null || echo)
else
    readonly BOLD="" RED="" GREEN="" YELLOW="" BLUE="" RESET=""
fi

log_info() {
    echo "${BLUE}${BOLD}[INFO]${RESET} $*"
}

log_success() {
    echo "${GREEN}${BOLD}[SUCCESS]${RESET} $*"
}

log_warning() {
    echo "${YELLOW}${BOLD}[WARNING]${RESET} $*" >&2
}

log_error() {
    echo "${RED}${BOLD}[ERROR]${RESET} $*" >&2
    exit 1
}

check_os_support() {
    case "$OS" in
        Linux|Darwin)
            return 0
            ;;
        *)
            log_error "Unsupported operating system: $OS. This script only supports Linux and macOS."
            ;;
    esac
}

add_common_uv_paths() {
    case "$OS" in
        Linux)
            if [[ -d "$HOME/.local/bin" && ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
                export PATH="$HOME/.local/bin:$PATH"
            fi
            ;;
        Darwin)
            if [[ -d "$HOME/.local/bin" && ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
                export PATH="$HOME/.local/bin:$PATH"
            fi
            for p in "$HOME/Library/Python/3."*"/bin"; do
                if [[ -d "$p" && ":$PATH:" != *":$p:"* ]]; then
                    export PATH="$p:$PATH"
                fi
            done
            ;;
    esac
}

ensure_uv_installed() {
    add_common_uv_paths

    if command -v uv &>/dev/null; then
        return 0
    fi

    log_info "uv not found. Installing uv automatically ($UV_INSTALL_METHOD)..."

    if [[ "$UV_INSTALL_METHOD" == "pip" ]]; then
        log_info "Installing uv with pip..."
        if ! command -v "$PYTHON" &>/dev/null; then
            log_error "$PYTHON not found, cannot install uv via pip."
        fi

        "$PYTHON" -m pip --version &>/dev/null || log_error "pip not available for $PYTHON"

        "$PYTHON" -m pip install --user uv || log_error "pip install uv failed"

        add_common_uv_paths

        if ! command -v uv &>/dev/null; then
            log_error "uv installed via pip but 'uv' command not found in PATH. Tried adding common directories. Please add the appropriate Python script directory to your PATH manually."
        fi
        log_info "uv installed via pip successfully: $(uv --version)"
    else
        if command -v curl &>/dev/null; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
        elif command -v wget &>/dev/null; then
            wget -qO- https://astral.sh/uv/install.sh | sh
        else
            log_error "Neither curl nor wget found. Please install curl or wget, or install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
        fi

        add_common_uv_paths

        if ! command -v uv &>/dev/null; then
            log_error "Failed to install uv. Please install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
        fi

        log_info "uv installed successfully: $(uv --version)"
    fi
}

setup_venv() {
    if [[ ! -d ".venv" ]]; then
        log_info "Creating virtual environment with uv (Python $UV_PYTHON_VERSION)..."
        uv venv --python "$UV_PYTHON_VERSION" --seed || \
            log_error "Failed to create virtual environment. Ensure Python $UV_PYTHON_VERSION is available (try: uv python install $UV_PYTHON_VERSION)"
    fi

    export PATH="$(pwd)/.venv/bin:$PATH"
    source .venv/bin/activate || log_error "Failed to activate virtual environment"
    log_info "Virtual environment ready: .venv (Python: $(python --version 2>&1))"
}

INSTALL_PACKAGES=false
INSTALL_MODEL=false
INSTALL_DATASET=false

MODEL_NAME=""
MODEL_SOURCE_PATH=""
MODEL_TARGET_DIR=""

DATASET_PATH=""
DATASET_NAME=""
DATASET_SPLIT="train"
DATASET_MAX_LENGTH=""
DATASET_CONFIG=""
DATASET_SHUFFLE=false
DATASET_SEED=42
DATASET_STREAMING=false
DATASET_INDENT=4
DATASET_OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"

usage() {
    cat << EOF
${BOLD}USAGE${RESET}
    $0 [OPTIONS]

${BOLD}MODES${RESET}
    (no arguments)               Install all Python packages
    --model MODEL                Install a model (currently only 'md')
    --dataset DATASET            Download a Hugging Face dataset

${BOLD}MODEL INSTALLATION OPTIONS${RESET}
    --model MODEL                Model name to install (required, only 'md' supported)
    --path PATH                  Local directory path to an existing model
                                 If not provided, clones from GitHub.

${BOLD}DATASET INSTALLATION OPTIONS${RESET}
    --dataset DATASET            Hugging Face dataset identifier (required)
                                 Example: 'metalearningnet/qwen3.5-metamathqa-cot'
    --name NAME                  Output filename without extension (required)
    --split SPLIT                Dataset split to download
    --config CONFIG              Configuration name for multi‑config datasets
    --max-length LENGTH          Limit number of examples
    --output-dir DIR             Directory to save the dataset

${BOLD}GENERAL OPTIONS${RESET}
    --help, -h                   Show this help message

${BOLD}ENVIRONMENT VARIABLES${RESET}
    UV_VLLM_VERSION              vLLM package version
    UV_INSTALL_METHOD            How to install uv if missing: 'standalone' or 'pip'

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                if [[ -n ${2:-} ]]; then
                    INSTALL_MODEL=true
                    MODEL_NAME="$2"
                    shift 2
                else
                    log_error "--model requires a value"
                fi
                ;;
            --path)
                if [[ -n ${2:-} ]]; then
                    if [[ "$INSTALL_MODEL" == false ]]; then
                        log_error "--path argument must follow --model argument"
                    fi
                    MODEL_SOURCE_PATH="$2"
                    shift 2
                else
                    log_error "--path requires a value"
                fi
                ;;
            --dataset)
                if [[ -n ${2:-} ]]; then
                    INSTALL_DATASET=true
                    DATASET_PATH="$2"
                    shift 2
                else
                    log_error "--dataset requires a value"
                fi
                ;;
            --name)
                if [[ -n ${2:-} ]]; then
                    DATASET_NAME="$2"
                    shift 2
                else
                    log_error "--name requires a value"
                fi
                ;;
            --split)
                if [[ -n ${2:-} ]]; then
                    DATASET_SPLIT="$2"
                    shift 2
                else
                    log_error "--split requires a value"
                fi
                ;;
            --max-length)
                if [[ -n ${2:-} ]]; then
                    if [[ ! "$2" =~ ^[0-9]+$ ]]; then
                        log_error "--max-length must be a positive integer"
                    fi
                    DATASET_MAX_LENGTH="$2"
                    shift 2
                else
                    log_error "--max-length requires a value"
                fi
                ;;
            --config)
                if [[ -n ${2:-} ]]; then
                    DATASET_CONFIG="$2"
                    shift 2
                else
                    log_error "--config requires a value"
                fi
                ;;
            --output-dir)
                if [[ -n ${2:-} ]]; then
                    DATASET_OUTPUT_DIR="$2"
                    shift 2
                else
                    log_error "--output-dir requires a value"
                fi
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1 (use --help for usage information)"
                ;;
            *)
                log_error "Unexpected argument: $1 (use --help for usage information)"
                ;;
        esac
    done
}

validate_arguments() {
    local mode_count=0
    [[ "${INSTALL_MODEL}" == "true" ]] && mode_count=$((mode_count + 1))
    [[ "${INSTALL_DATASET:-}" == "true" ]] && mode_count=$((mode_count + 1))

    if [[ $mode_count -gt 1 ]]; then
        log_error "Cannot combine --model and --dataset in the same command. Please choose one mode."
    fi

    if [[ $mode_count -eq 0 ]]; then
        INSTALL_PACKAGES=true
    fi

    if [[ "${INSTALL_DATASET:-}" == true ]]; then
        if [[ -z "${DATASET_PATH:-}" ]]; then
            log_error "Dataset path is required when using --dataset"
        fi
        if [[ -z "${DATASET_NAME:-}" ]]; then
            log_error "--name is required when using --dataset"
        fi
    fi

    if [[ "${INSTALL_MODEL:-}" == true ]]; then
        if [[ -z "${MODEL_NAME:-}" ]]; then
            log_error "Model name is required when using --model"
        fi
        if [[ "$MODEL_NAME" != "md" ]]; then
            log_error "Currently only 'md' model is supported. You specified: $MODEL_NAME"
        fi

        if [[ -n "${MODEL_SOURCE_PATH:-}" ]]; then
            local check_path="$MODEL_SOURCE_PATH"
            if [[ "$check_path" == ~* ]]; then
                check_path="${HOME}${check_path:1}"
            fi
            if [[ "$check_path" == /* ]] || [[ "$MODEL_SOURCE_PATH" == ~* ]]; then
                if [[ ! -e "$check_path" ]]; then
                    log_error "Specified --path does not exist: $MODEL_SOURCE_PATH (resolved: $check_path)"
                elif [[ ! -d "$check_path" ]]; then
                    log_error "Specified --path is not a directory: $MODEL_SOURCE_PATH (resolved: $check_path)"
                elif [[ ! -r "$check_path" ]]; then
                    log_error "Specified --path is not readable (permission denied): $MODEL_SOURCE_PATH (resolved: $check_path)"
                fi
            fi
        fi
    fi
}

install_pytorch() {
    log_info "Installing PyTorch with CUDA version $TORCH_CUDA_VERSION..."
    local index_url="https://download.pytorch.org/whl/$TORCH_CUDA_VERSION"
    uv pip install "torch==$TORCH_VERSION" torchvision torchaudio --index-url "$index_url" \
        || log_error "Failed to install PyTorch from $index_url"

    log_info "Verifying PyTorch installation..."
    if ! uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        log_error "PyTorch CUDA test failed. This often indicates a missing or incompatible NCCL library."
    fi
    log_info "PyTorch installed and CUDA is available."
}

install_packages() {
    local package_list=(
        absl-py accelerate aiohappyeyeballs aiohttp aiosignal
        altair annotated-types anyio asttokens async-timeout
        attrs bitsandbytes blessed blinker cachetools
        certifi chanfig charset-normalizer click coloredlogs
        comm contourpy cycler danling datasets
        debugpy decorator dill distro docker-pycreds
        einops exceptiongroup executing filelock fonttools
        frozenlist fsspec gitdb GitPython gpustat
        grpcio h11 httpcore httpx huggingface-hub
        humanfriendly idna importlib_metadata importlib_resources ipykernel
        ipython jedi Jinja2 jiter joblib
        jsonlines jsonschema jsonschema-specifications jupyter_client jupyter_core
        kiwisolver lazy-imports Markdown markdown-it-py MarkupSafe
        matplotlib matplotlib-inline mdurl mpmath multidict
        multimolecule multiprocess narwhals nest-asyncio networkx
        nltk numpy openai optimum packaging
        pandas parso peft pexpect pillow
        platformdirs prompt_toolkit protobuf psutil ptyprocess
        pure_eval pyarrow pydantic pydantic_core pydeck
        Pygments pyparsing python-dateutil pytz PyYAML
        pyzmq referencing regex requests rich
        rpds-py safetensors scikit-learn scipy seaborn
        sentence-transformers sentencepiece sentry-sdk seqeval setproctitle
        six smmap sniffio stack-data streamlit
        StrEnum sympy tenacity tensorboard tensorboard-data-server
        threadpoolctl tokenizers toml tornado tqdm traitlets
        triton typing_extensions tzdata urllib3 wandb
        watchdog wcwidth Werkzeug xxhash yarl zipp trl
        alfworld h5py arc-agi
    )

    log_info "Installing Python packages..."
    if uv pip install "${package_list[@]}"; then
        log_info "All packages installed successfully"
    else
        log_error "Failed to install packages"
    fi

    log_info "Installing vLLM version ${UV_VLLM_VERSION}..."
    export UV_TORCH_BACKEND=auto
    uv pip install "vllm==${UV_VLLM_VERSION}" || log_warning "vLLM installation failed"
    uv pip install -U transformers || log_warning "Failed to upgrade Transformers"

    log_info "Verifying ALFWorld installation..."
    if ! uv run python -c "import alfworld" 2>/dev/null; then
        log_error "ALFWorld package not properly installed. Please check the installation logs."
    fi

    if ! command -v alfworld-download &>/dev/null; then
        log_error "alfworld-download command not found. ALFWorld installation appears incomplete."
    fi

    log_info "Downloading ALFWorld data..."
    if alfworld-download; then
        log_info "ALFWorld data downloaded successfully"
    else
        log_error "ALFWorld data download failed. Cannot proceed without required game files."
    fi
}

install_model() {
    log_info "Installing $MODEL_NAME model to $MODEL_TARGET_DIR..."

    if [[ "$MODEL_NAME" != "md" ]]; then
        log_error "Currently only 'md' model is supported"
    fi

    mkdir -p "$(dirname "$MODEL_TARGET_DIR")"

    if [[ -d "$MODEL_TARGET_DIR" ]]; then
        log_warning "Removing existing ${MODEL_NAME} directory at $MODEL_TARGET_DIR"
        rm -rf "$MODEL_TARGET_DIR"
    fi

    if [[ -n "$MODEL_SOURCE_PATH" ]]; then
        if [[ -d "$MODEL_SOURCE_PATH" ]]; then
            log_info "Copying $MODEL_NAME model from $MODEL_SOURCE_PATH to $MODEL_TARGET_DIR"
            cp -r "$MODEL_SOURCE_PATH" "$MODEL_TARGET_DIR" || \
                log_error "Failed to copy model from $MODEL_SOURCE_PATH to $MODEL_TARGET_DIR"
        else
            log_error "Local $MODEL_NAME path $MODEL_SOURCE_PATH does not exist or is not a directory"
        fi
    else
        log_info "Cloning $MODEL_NAME model from GitHub to $MODEL_TARGET_DIR"
        if git clone https://github.com/metalearningnet/md.git "$MODEL_TARGET_DIR"; then
            log_info "$MODEL_NAME model cloned successfully"
        else
            log_error "Failed to clone $MODEL_NAME model from GitHub"
        fi
    fi

    if [[ ! -d "$MODEL_TARGET_DIR" ]]; then
        log_error "Failed to install $MODEL_NAME model"
    fi

    local model_install_script="${MODEL_TARGET_DIR}/install.sh"
    if [[ -f "$model_install_script" ]]; then
        log_info "Found model-specific installation script: $model_install_script"
        log_info "Executing model installation script..."
        if bash "$model_install_script"; then
            log_info "Model installation script completed successfully"
        else
            log_error "Model installation script failed: $model_install_script"
        fi
    else
        log_warning "No model-specific install.sh found at $model_install_script (optional, continuing)"
    fi

    log_info "$MODEL_NAME model successfully installed to $MODEL_TARGET_DIR"
}

install_dataset() {
    log_info "Downloading dataset..."

    if [[ ! -f "$DATASET_LOADER_SCRIPT" ]]; then
        log_error "Dataset loader script not found: $DATASET_LOADER_SCRIPT"
    fi

    log_info "Dataset configuration:"
    log_info "  Dataset identifier: $DATASET_PATH"
    log_info "  Output filename: ${DATASET_NAME}_${DATASET_SPLIT}.json"
    log_info "  Target split: $DATASET_SPLIT"
    log_info "  Output directory: $DATASET_OUTPUT_DIR"
    [[ -n "$DATASET_MAX_LENGTH" ]] && log_info "  Maximum examples: $DATASET_MAX_LENGTH"
    [[ -n "$DATASET_CONFIG" ]] && log_info "  Configuration: $DATASET_CONFIG"
    [[ "$DATASET_SHUFFLE" == true ]] && log_info "  Shuffle: enabled (seed: $DATASET_SEED)"
    [[ "$DATASET_STREAMING" == true ]] && log_info "  Streaming mode: enabled"

    local CMD_ARGS=(
        "--dataset" "$DATASET_PATH"
        "--name" "$DATASET_NAME"
        "--split" "$DATASET_SPLIT"
        "--output_dir" "$DATASET_OUTPUT_DIR"
        "--indent" "$DATASET_INDENT"
        "--seed" "$DATASET_SEED"
    )

    [[ -n "$DATASET_MAX_LENGTH" ]] && CMD_ARGS+=("--max_length" "$DATASET_MAX_LENGTH")
    [[ -n "$DATASET_CONFIG" ]] && CMD_ARGS+=("--config_name" "$DATASET_CONFIG")
    [[ "$DATASET_SHUFFLE" == true ]] && CMD_ARGS+=("--shuffle")
    [[ "$DATASET_STREAMING" == true ]] && CMD_ARGS+=("--streaming")

    log_info "Executing: uv run python $DATASET_LOADER_SCRIPT ${CMD_ARGS[*]}"

    if uv run python "$DATASET_LOADER_SCRIPT" "${CMD_ARGS[@]}"; then
        local OUTPUT_FILE="${DATASET_OUTPUT_DIR}/${DATASET_NAME}_${DATASET_SPLIT}.json"
        if [[ ! -f "$OUTPUT_FILE" ]]; then
            log_warning "Dataset download completed but output file not found."
            log_warning "Expected: $OUTPUT_FILE"
        fi
    else
        log_error "Failed to download dataset. Check the error messages above."
    fi
}

main() {
    check_os_support
    ensure_uv_installed
    parse_arguments "$@"
    validate_arguments
    setup_venv

    if [[ "${INSTALL_MODEL}" == true ]]; then
        MODEL_TARGET_DIR="./targets/${MODEL_NAME}"
    fi

    if [[ "$INSTALL_PACKAGES" == true ]]; then
        install_pytorch
        install_packages
    elif [[ "$INSTALL_MODEL" == true ]]; then
        install_model
        install_pytorch
        install_packages
    elif [[ "$INSTALL_DATASET" == true ]]; then
        install_dataset
    fi

    show_summary
}

show_summary() {
    if [[ "$INSTALL_DATASET" == true ]]; then
        log_success "Dataset installation completed successfully!"
    elif [[ "$INSTALL_MODEL" == true ]]; then
        log_success "Model installation completed successfully!"
    elif [[ "$INSTALL_PACKAGES" == true ]]; then
        log_success "Package installation completed successfully!"
    fi
}

main "$@"
