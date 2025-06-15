# --------------------------------------------------
# Required Third-Party Libraries for ML/NLP projects
# --------------------------------------------------

# 1. tiktoken: Used for OpenAI-compatible tokenization
try:
    import tiktoken
except ImportError:
    raise ImportError("Please install 'tiktoken' with `pip install tiktoken`")

# 2. gdown: Used for downloading from Google Drive
try:
    import gdown
except ImportError:
    raise ImportError("Please install 'gdown' with `pip install gdown`")

# 3. transformers: Hugging Face Transformers library
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    raise ImportError("Please install 'transformers' with `pip install transformers`")

# 4. huggingface_hub: Direct access to Hugging Face model hub
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("Please install 'huggingface_hub' with `pip install huggingface_hub`")

# 5. sentencepiece: Needed for models like T5, BERT multilingual, etc.
try:
    import sentencepiece
except ImportError:
    raise ImportError("Please install 'sentencepiece' with `pip install sentencepiece`")

# 6. blobfile: File I/O for remote and local data (used by some datasets)
try:
    import blobfile as bf
except ImportError:
    raise ImportError("Please install 'blobfile' with `pip install blobfile`")

# 7. safetensors: Safer and faster tensor serialization format
try:
    import safetensors
    from safetensors import safe_open
    from safetensors.torch import load_file
except ImportError:
    raise ImportError("Please install 'safetensors>=0.4.4' with `pip install safetensors>=0.4.4`")

# -------------------------------
# Optional: Log that all imports are successful
# -------------------------------
from logger import setup_logger
logger = setup_logger("dependencies")

logger.info("All required libraries imported successfully.")
