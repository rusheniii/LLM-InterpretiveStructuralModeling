import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock

from llmism.ISMBaseStep import ISMBaseStep


class RecordingStep(ISMBaseStep):
    async def process(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


class TestISMBaseStep(IsolatedAsyncioTestCase):
    def test_pickle_results_writes_pickled_data(self) -> None:
        expected = {"answer": 42}
        step = ISMBaseStep(Mock(), Path("checkpoint.bin"))

        with TemporaryDirectory() as directory:
            data_file = Path(directory) / "result.bin"

            step.pickle_results(expected, data_file)

            with data_file.open("rb") as file:
                self.assertEqual(pickle.load(file), expected)

    def test_unpickle_results_reads_pickled_data(self) -> None:
        expected = {"answer": 42}
        step = ISMBaseStep(Mock(), Path("checkpoint.bin"))

        with TemporaryDirectory() as directory:
            data_file = Path(directory) / "result.bin"
            with data_file.open("wb") as file:
                pickle.dump(expected, file)

            result = step.unpickle_results(data_file)

            self.assertEqual(result, expected)

    async def test_process_and_save_returns_cached_checkpoint(self) -> None:
        expected = {"cached": True}

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.bin"
            with checkpoint.open("wb") as file:
                pickle.dump(expected, file)
            step = RecordingStep(Mock(), checkpoint)
            step.process = AsyncMock()

            result = await step.process_and_save("input")

            self.assertEqual(result, expected)
            step.process.assert_not_called()

    async def test_process_and_save_processes_and_persists_missing_checkpoint(
        self,
    ) -> None:
        expected = {"args": ("input",), "kwargs": {"flag": True}}

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.bin"
            step = RecordingStep(Mock(), checkpoint)

            result = await step.process_and_save("input", flag=True)

            self.assertEqual(result, expected)
            with checkpoint.open("rb") as file:
                self.assertEqual(pickle.load(file), expected)


class TestRecordingStep(TestCase):
    def test_recording_step_inherits_base_step(self) -> None:
        expected = True

        result = issubclass(RecordingStep, ISMBaseStep)

        self.assertEqual(result, expected)
