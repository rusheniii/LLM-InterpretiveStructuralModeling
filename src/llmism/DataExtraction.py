from pathlib import Path
import docling.document_converter as dc
import tiktoken
import re
import html
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling.chunking import HybridChunker

from .LLMClient import LLMClient
from .ISMBaseStep import ISMBaseStep

import logging

logger = logging.getLogger(__name__)


class DataExtractionStepChunker(ISMBaseStep):
    def __init__(self, client: LLMClient, checkpoint: Path):
        super().__init__(client, checkpoint)
        self.converter = dc.DocumentConverter()
        tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model("gpt-4o"),
            max_tokens=8191,
        )
        self.chunker = HybridChunker(tokenizer=tokenizer)

    def _sanitize_text_chunk(self, chunk: str) -> str:
        text = chunk.encode('ascii', 'ignore').decode()
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = html.escape(text)
        return text

    async def process_pdf(self, pdf):
        doc = self.converter.convert(pdf).document
        chunks = self.chunker.chunk(dl_doc=doc)
        text_chunks = []
        for chunk in chunks:
            chunk_text = self._sanitize_text_chunk(chunk.text)
            text_chunks.append(chunk_text)
        return text_chunks

    async def process(self, pdfs):
        chunks = []
        for pdf in pdfs:
            name = str(pdf)
            logger.info("Processing %s" % name)
            text_chunks = await self.process_pdf(name)
            chunks.extend(text_chunks)
        return chunks
