import os
from pathlib import Path
import pickle
from .LLMClient import LLMClient


class ISMBaseStep(object):
    def __init__(self, client: LLMClient, checkpoint: Path = None):
        self.client = client
        self.checkpoint = checkpoint

    def pickle_results(self, data, data_file: str):
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

    def unpickle_results(self, data_file: str):
        with open(data_file, 'rb') as f:
            return pickle.load(f)

    async def process_and_save(self, *args, **kwargs):
        print('Executing step')
        if self.checkpoint and os.path.exists(self.checkpoint):
            return self.unpickle_results(self.checkpoint)
        else:
            output = await self.process(*args, **kwargs)
            if self.checkpoint:
                self.pickle_results(output, self.checkpoint)
            return output
