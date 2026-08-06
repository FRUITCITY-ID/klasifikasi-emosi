from __future__ import annotations

import os
import unittest
from pathlib import Path

import config


class ProjectPathTests(unittest.TestCase):
    def test_relative_path_is_anchored_to_project(self) -> None:
        resolved = config.resolve_project_path("models/example.pt")

        self.assertEqual(resolved, config.PROJECT_DIR / "models" / "example.pt")

    def test_absolute_path_is_preserved(self) -> None:
        absolute = Path("C:/model-store/example.pt")

        self.assertEqual(config.resolve_project_path(absolute), absolute)

    def test_default_checkpoint_is_portable(self) -> None:
        if "SIPEMO_CHECKPOINT" not in os.environ:
            self.assertEqual(
                config.CHECKPOINT_PATH,
                config.PROJECT_DIR / "models" / "best_indobert_base.pt",
            )
