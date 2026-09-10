# -*- coding: utf-8 -*-
"""Tests for the multi-worker run status / row-count sanity tool."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from udwm.scripts.run_status import main

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "_probe" / "status_tests"


@pytest.fixture()
def workdir(request):
    d = TMP / request.node.name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_complete_split_style_partials(workdir, capsys):
    out = workdir / "study.json"
    for s in (0, 1):
        _write(workdir / "study_seed{}.partial.json".format(s),
               {"rows": [{"seed": s, "variant": v} for v in ("ordinary", "hybrid")]})
    rc = main(["--out", str(out), "--expect-seeds", "2", "--arms", "ordinary", "hybrid"])
    text = capsys.readouterr().out
    assert rc == 0
    assert "COMPLETE" in text
    assert "2 seeds x 2 arms = 4 rows" in text


def test_incomplete_reports_missing_pairs(workdir, capsys):
    out = workdir / "study.json"
    _write(workdir / "study_seed0.partial.json",
           {"rows": [{"seed": 0, "variant": "ordinary"}]})
    main(["--out", str(out), "--expect-seeds", "2", "--arms", "ordinary", "hybrid"])
    text = capsys.readouterr().out
    assert "INCOMPLETE" in text
    assert "s0:hybrid" in text and "s1:ordinary" in text


def test_probe_single_row_partials_are_counted(workdir, capsys):
    out = workdir / "crn_bias_smoke.json"
    for arm in ("identified_eq", "hybrid"):
        _write(workdir / "crn_bias_smoke_s0_{}.partial.json".format(arm),
               {"seed": 0, "variant": arm, "final_return": 0.0})
    main(["--out", str(out), "--expect-seeds", "1", "--arms", "identified_eq", "hybrid"])
    text = capsys.readouterr().out
    assert "COMPLETE" in text
    assert "2 rows" in text


def test_canonical_file_read_and_summarizer_choice(workdir, capsys):
    out = workdir / "walker_study.json"
    _write(out, {"protocol": {"variants": ["ordinary"]},
                 "rows": [{"seed": 0, "variant": "ordinary"}]})
    main(["--out", str(out), "--expect-seeds", "1"])
    text = capsys.readouterr().out
    assert "COMPLETE" in text
    assert "summarize_walker2d_payoff" in text


def test_truncated_partial_does_not_crash(workdir, capsys):
    out = workdir / "study.json"
    (workdir / "study_seed0.partial.json").write_text('{"rows": [', encoding="utf-8")
    _write(workdir / "study_seed1.partial.json",
           {"rows": [{"seed": 1, "variant": "ordinary"}]})
    main(["--out", str(out), "--expect-seeds", "2", "--arms", "ordinary"])
    text = capsys.readouterr().out
    assert "INCOMPLETE" in text          # seed 0 is not counted as present
    assert "1 rows" in text
