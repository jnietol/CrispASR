"""Tier 0: dev-box smoke test for the regression driver.

No model downloads, no binary invocations, no network. Just:

  - manifest.json schema and shape (every backend has the required
    keys, types are right, no obvious typos).
  - parse_diff_stdout()       — feed it canned crispasr-diff output,
                                 assert it pulls cos_min for each stage.
  - evaluate_stage_thresholds() — the threshold-compare logic that
                                 decides per-stage PASS/FAIL/missing.

Runs in well under a second. Catches manifest typos and parser
regressions in PR CI without needing a built binary or HF auth.

Usage:

  python -m unittest tests/regression/test_driver_smoke.py

or via pytest if available:

  pytest tests/regression/test_driver_smoke.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `import run_one` works

import run_one  # noqa: E402


# Sample crispasr-diff stdout — actual capture from the parakeet JA seed,
# with ANSI codes stripped. Tweak when crispasr-diff's output format
# changes.
SAMPLE_DIFF_STDOUT = """\
crispasr-diff: audio 224964 samples (14.06s), reference /path/ref.gguf, backend parakeet
[FAIL] mel_spectrogram        shape=[80,1407]        cos_min=0.951084  cos_mean=0.999388  max_abs=2.23e+00  rms=3.67e-02
[PASS] encoder_output         shape=[1024,176]       cos_min=0.999594  cos_mean=0.999975  max_abs=3.13e-02  rms=1.56e-03
[PASS] encoder_output_ref_mel shape=[1024,176]       cos_min=0.999807  cos_mean=0.999988  max_abs=2.28e-02  rms=1.08e-03

summary: 2 pass, 1 fail, 0 skip (cos threshold 0.999)
"""


class ManifestSchemaTests(unittest.TestCase):
    """The manifest is the source of truth — typos here corrupt every
    nightly run silently. Lock the shape.
    """

    @classmethod
    def setUpClass(cls):
        with (HERE / "manifest.json").open() as f:
            cls.manifest = json.load(f)

    def test_version_pinned(self):
        self.assertEqual(self.manifest.get("version"), 1)

    def test_fixtures_block_pinned(self):
        fx = self.manifest.get("fixtures", {})
        self.assertIsInstance(fx, dict)
        self.assertIn("repo", fx)
        self.assertIn("revision", fx)
        # "main" is technically valid but defeats the entire upstream-
        # safety guarantee. Block it explicitly so a sloppy revert
        # can't slip in.
        self.assertNotEqual(
            fx["revision"], "main",
            "fixtures.revision must be pinned to a commit SHA, not a branch name "
            "(see tests/regression/README.md for why).",
        )
        self.assertRegex(
            fx["revision"], r"^[0-9a-f]{7,40}$",
            "fixtures.revision should look like a git SHA (7-40 hex chars).",
        )

    def test_backends_present(self):
        self.assertIn("backends", self.manifest)
        self.assertIsInstance(self.manifest["backends"], list)
        self.assertGreater(len(self.manifest["backends"]), 0)

    def test_backend_entries_have_required_keys(self):
        # Keys required by every entry regardless of skip_diff.
        always_required = {"name", "backend_id", "gguf", "expected_transcript"}
        # Keys required only for full diff entries (skip_diff absent or false).
        diff_required = {"fixture_ref_path", "diff_thresholds"}
        gguf_required = {"repo", "revision", "file"}
        names_seen: set[str] = set()
        for entry in self.manifest["backends"]:
            missing = always_required - set(entry)
            self.assertEqual(missing, set(),
                f"backend missing keys: {missing} in {entry.get('name', '?')}")
            self.assertNotIn(entry["name"], names_seen,
                f"duplicate backend name: {entry['name']}")
            names_seen.add(entry["name"])
            self.assertEqual(
                gguf_required - set(entry["gguf"]), set(),
                f"{entry['name']}.gguf missing keys",
            )
            # Pinned revision must be a SHA, not a branch.
            rev = entry["gguf"]["revision"]
            self.assertRegex(
                rev, r"^[0-9a-f]{7,40}$",
                f"{entry['name']}.gguf.revision must be a SHA, got {rev!r}",
            )
            skip_diff = entry.get("skip_diff", False)
            if not skip_diff:
                # Full diff entry: fixture_ref_path + diff_thresholds required.
                missing_diff = diff_required - set(entry)
                self.assertEqual(missing_diff, set(),
                    f"{entry['name']}: diff entry missing keys {missing_diff} "
                    f"(set skip_diff=true if no ref dump exists yet)")
                self.assertIsInstance(entry["diff_thresholds"], dict)
                self.assertGreater(len(entry["diff_thresholds"]), 0,
                    f"{entry['name']} has empty diff_thresholds — at least one stage must be tracked")
                for stage, threshold in entry["diff_thresholds"].items():
                    self.assertIsInstance(threshold, (int, float),
                        f"{entry['name']}.{stage} threshold must be numeric")
                    self.assertTrue(
                        0.0 <= threshold <= 1.0,
                        f"{entry['name']}.{stage} threshold {threshold} out of [0,1]",
                    )
            else:
                # Transcript-only entry: diff_thresholds and fixture_ref_path
                # must NOT be present (forces the maintainer to graduate it
                # to a full diff entry once the ref dump is baked).
                self.assertNotIn("diff_thresholds", entry,
                    f"{entry['name']}: skip_diff=true but diff_thresholds is also set — "
                    f"remove skip_diff once the ref dump is in cstr/crispasr-regression-fixtures")
                self.assertNotIn("fixture_ref_path", entry,
                    f"{entry['name']}: skip_diff=true but fixture_ref_path is also set — "
                    f"remove skip_diff once the ref dump is in cstr/crispasr-regression-fixtures")

    def test_sample_source_declared(self):
        """Every backend must declare its sample source: either a
        repo-relative `sample` (in-tree, must exist) or a
        `fixture_sample_path` pulled from the fixtures HF repo at
        the pinned revision (the recommended path for new entries).
        """
        for entry in self.manifest["backends"]:
            has_in_repo = "sample" in entry
            has_hf = "fixture_sample_path" in entry
            self.assertTrue(
                has_in_repo or has_hf,
                f"{entry['name']}: neither `sample` (in-repo) nor "
                f"`fixture_sample_path` (HF) declared",
            )
            if has_in_repo:
                sample = HERE.parent.parent / entry["sample"]
                self.assertTrue(
                    sample.exists(),
                    f"{entry['name']}: in-repo sample {sample} not "
                    f"present; either drop the field or check the WAV in",
                )


class DiffParserTests(unittest.TestCase):
    """parse_diff_stdout() — given a captured crispasr-diff stdout,
    pull cos_min per stage. Don't care about [PASS]/[FAIL] verdict
    (we apply manifest thresholds separately).
    """

    def test_basic_capture(self):
        stages = run_one.parse_diff_stdout(SAMPLE_DIFF_STDOUT)
        self.assertEqual(
            set(stages),
            {"mel_spectrogram", "encoder_output", "encoder_output_ref_mel"},
        )
        # Spot-check values land where the line says they should.
        self.assertAlmostEqual(stages["mel_spectrogram"], 0.951084, places=6)
        self.assertAlmostEqual(stages["encoder_output"], 0.999594, places=6)
        self.assertAlmostEqual(stages["encoder_output_ref_mel"], 0.999807, places=6)

    def test_handles_empty_stdout(self):
        self.assertEqual(run_one.parse_diff_stdout(""), {})

    def test_handles_garbage_lines(self):
        """Output without `[PASS]/[FAIL]` lines must yield {}."""
        garbage = "loading model...\nstage A failed (oops)\nfinished\n"
        self.assertEqual(run_one.parse_diff_stdout(garbage), {})

    def test_scientific_notation(self):
        """The regex must accept scientific-notation cos_min values
        (rare but produced when stages are numerically degenerate).
        """
        line = "[FAIL] foo shape=[1] cos_min=1.0e-05 cos_mean=2.5e-05"
        stages = run_one.parse_diff_stdout(line)
        self.assertIn("foo", stages)
        self.assertAlmostEqual(stages["foo"], 1.0e-05, places=10)

    def test_negative_cos_min(self):
        """`cos_min` can be negative when the captured stage anti-
        correlates with the reference (a real failure mode worth
        catching). Regex must allow the sign.
        """
        line = "[FAIL] anti shape=[8,1] cos_min=-0.42 cos_mean=-0.3"
        stages = run_one.parse_diff_stdout(line)
        self.assertAlmostEqual(stages["anti"], -0.42, places=6)


class ThresholdEvaluationTests(unittest.TestCase):
    """evaluate_stage_thresholds() — partition cos_min map into
    pass / fail / missing / extras given the manifest's thresholds.
    """

    THRESHOLDS = {
        "encoder_output": 0.999,
        "encoder_output_ref_mel": 0.999,
        "mel_spectrogram": 0.95,
    }

    def test_all_pass(self):
        stages = {
            "encoder_output": 0.9999,
            "encoder_output_ref_mel": 0.9999,
            "mel_spectrogram": 0.96,
        }
        p, f, m, e = run_one.evaluate_stage_thresholds(stages, self.THRESHOLDS)
        self.assertEqual(len(p), 3)
        self.assertEqual(f, [])
        self.assertEqual(m, [])
        self.assertEqual(e, [])

    def test_threshold_strict_below_fails(self):
        """A stage at exactly `threshold` PASSES; below FAILS."""
        stages = {
            "encoder_output": 0.999,           # == threshold → PASS
            "encoder_output_ref_mel": 0.9989,  # < threshold → FAIL
            "mel_spectrogram": 0.95,           # == threshold → PASS
        }
        p, f, m, e = run_one.evaluate_stage_thresholds(stages, self.THRESHOLDS)
        passes = {x[0] for x in p}
        fails = {x[0] for x in f}
        self.assertIn("encoder_output", passes)
        self.assertIn("mel_spectrogram", passes)
        self.assertIn("encoder_output_ref_mel", fails)

    def test_missing_stage(self):
        """If the manifest asks for a stage the diff harness didn't
        emit, it shows up in `missing` (not silently in `passes`).
        """
        stages = {"encoder_output": 0.9999}
        p, f, m, e = run_one.evaluate_stage_thresholds(stages, self.THRESHOLDS)
        self.assertEqual(set(m), {"encoder_output_ref_mel", "mel_spectrogram"})
        self.assertEqual(len(p), 1)
        self.assertEqual(f, [])

    def test_extra_stage(self):
        """A stage in the diff output but not in thresholds should be
        flagged as an extra (so a maintainer can add it intentionally).
        """
        stages = {
            "encoder_output": 0.9999,
            "encoder_output_ref_mel": 0.9999,
            "mel_spectrogram": 0.96,
            "some_new_stage": 0.42,  # extra
        }
        p, f, m, e = run_one.evaluate_stage_thresholds(stages, self.THRESHOLDS)
        self.assertEqual(e, [("some_new_stage", 0.42)])

    def test_end_to_end_against_canned_diff_output(self):
        """Wire parse_diff_stdout → evaluate_stage_thresholds against
        the canned sample; assert the partition matches what the
        seed backend (parakeet JA) actually produces.
        """
        stages = run_one.parse_diff_stdout(SAMPLE_DIFF_STDOUT)
        p, f, m, e = run_one.evaluate_stage_thresholds(stages, self.THRESHOLDS)
        passing = {x[0] for x in p}
        # All three stages clear their (lenient) thresholds. The diff
        # harness's own `[FAIL] mel_spectrogram` verdict is ignored
        # because manifest mel threshold is 0.95, not 0.999.
        self.assertEqual(
            passing,
            {"encoder_output", "encoder_output_ref_mel", "mel_spectrogram"},
        )
        self.assertEqual(f, [])
        self.assertEqual(m, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
