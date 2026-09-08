from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src").exists():
    ROOT = Path.cwd()
SRC = ROOT / "src"

for path in (ROOT, SRC):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)


try:
    import docling.document_converter
except ModuleNotFoundError:
    docling = ModuleType("docling")
    document_converter = ModuleType("docling.document_converter")
    document_converter.DocumentConverter = object
    docling.document_converter = document_converter
    sys.modules["docling"] = docling
    sys.modules["docling.document_converter"] = document_converter

    docling_core = ModuleType("docling_core")
    transforms = ModuleType("docling_core.transforms")
    chunker = ModuleType("docling_core.transforms.chunker")
    tokenizer = ModuleType("docling_core.transforms.chunker.tokenizer")
    openai = ModuleType("docling_core.transforms.chunker.tokenizer.openai")
    openai.OpenAITokenizer = object
    docling_chunking = ModuleType("docling.chunking")
    docling_chunking.HybridChunker = object
    docling_core.transforms = transforms
    transforms.chunker = chunker
    chunker.tokenizer = tokenizer
    tokenizer.openai = openai
    docling.chunking = docling_chunking
    sys.modules.update(
        {
            "docling_core": docling_core,
            "docling_core.transforms": transforms,
            "docling_core.transforms.chunker": chunker,
            "docling_core.transforms.chunker.tokenizer": tokenizer,
            "docling_core.transforms.chunker.tokenizer.openai": openai,
            "docling.chunking": docling_chunking,
        }
    )


try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = ModuleType("tiktoken")
    tiktoken.encoding_for_model = lambda model: object()
    sys.modules["tiktoken"] = tiktoken
