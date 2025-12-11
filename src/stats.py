from typing import List
from textwrap import dedent

from pydantic import BaseModel


class Stats(BaseModel):
    start_time: float = 0
    end_time: float = 0
    sample_start_times: List[float] = []
    sample_end_times: List[float] = []
    prefill_tokens: int = 0
    generated_tokens: int = 0

    @property
    def run_time(self):
        return self.end_time - self.start_time

    def print(self):
        assert len(self.sample_start_times) == len(self.sample_end_times)
        n = len(self.sample_start_times)
        print(
            dedent(
                f"""
                Run time: {round(self.run_time, 2)}
                Prefill tokens: {self.prefill_tokens} tok
                Generated tokens: {self.generated_tokens} tok, {self.generated_tokens / self.run_time} tok/s
                Per sample latency from global start: {sum(self.sample_end_times) / n - self.start_time} s
                Per sample latency from sample start: {sum([e - s for e, s in zip(self.sample_end_times, self.sample_start_times)]) / n}
                """
            )
        )
