import csv
from pathlib import Path


class Loader:
    def __init__(self, path: str):
        self.path = Path(path)
        self.headers: list[str] = []
        self.rows: list[list[str]] = []
        self._load()

    def _load(self):
        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            self.headers = next(reader)
            for row in reader:
                if row and any(cell.strip() for cell in row):
                    self.rows.append(row)

    @property
    def feature_count(self) -> int:
        return len(self.headers) - 1 if len(self.headers) > 1 else len(self.headers)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return f"Loader(features={self.feature_count}, rows={self.row_count})"