from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

import llmism.DataExtraction as data_extraction_module
from llmism.DataExtraction import DataExtractionStepChunker
class TestDataExtractionStepChunker(IsolatedAsyncioTestCase):
    def test_init_configures_converter_tokenizer_and_chunker(self) -> None:
        client = Mock()
        checkpoint = Path("checkpoint.bin")
        expected_converter = Mock()
        expected_encoding = Mock()
        expected_tokenizer = Mock()
        expected_chunker = Mock()

        with (
            patch.object(data_extraction_module.dc, "DocumentConverter") as converter,
            patch.object(data_extraction_module.tiktoken, "encoding_for_model") as enc,
            patch.object(data_extraction_module, "OpenAITokenizer") as tokenizer,
            patch.object(data_extraction_module, "HybridChunker") as chunker,
        ):
            converter.return_value = expected_converter
            enc.return_value = expected_encoding
            tokenizer.return_value = expected_tokenizer
            chunker.return_value = expected_chunker

            result = DataExtractionStepChunker(client, checkpoint)

        self.assertIs(result.client, client)
        self.assertEqual(result.checkpoint, checkpoint)
        self.assertIs(result.converter, expected_converter)
        self.assertIs(result.chunker, expected_chunker)
        enc.assert_called_once_with("gpt-4o")
        tokenizer.assert_called_once_with(
            tokenizer=expected_encoding,
            max_tokens=8191,
        )
        chunker.assert_called_once_with(tokenizer=expected_tokenizer)

    def test_sanitize_text_chunk_removes_controls_and_escapes_html(self) -> None:
        expected = "AB&lt;tag&gt;"
        step = DataExtractionStepChunker.__new__(DataExtractionStepChunker)

        result = step._sanitize_text_chunk("A\x00B<tag>\u2603")

        self.assertEqual(result, expected)

    async def test_process_pdf_converts_chunks_to_sanitized_text(self) -> None:
        expected = ["safe&lt;text&gt;"]
        step = DataExtractionStepChunker.__new__(DataExtractionStepChunker)
        document = Mock(name="document")
        step.converter = Mock(
            convert=Mock(return_value=SimpleNamespace(document=document))
        )
        step.chunker = Mock(
            chunk=Mock(return_value=[SimpleNamespace(text="safe<text>")])
        )

        result = await step.process_pdf("paper.pdf")

        self.assertEqual(result, expected)
        step.converter.convert.assert_called_once_with("paper.pdf")
        step.chunker.chunk.assert_called_once_with(dl_doc=document)

    async def test_process_collects_chunks_from_each_pdf(self) -> None:
        expected = ["first", "second"]
        step = DataExtractionStepChunker.__new__(DataExtractionStepChunker)
        step.process_pdf = AsyncMock(
            side_effect=[
                ["first"],
                ["second"],
            ]
        )

        result = await step.process([Path("a.pdf"), Path("b.pdf")])

        self.assertEqual(result, expected)
        self.assertEqual(step.process_pdf.await_args_list[0].args, ("a.pdf",))
        self.assertEqual(step.process_pdf.await_args_list[1].args, ("b.pdf",))
