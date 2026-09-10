# -*- coding: utf-8 -*-
"""Tests for the shared race-safe additive merge (udwm/utils/atomic_merge.py).

The bug these pin: two invocations that ran the same study into one ``--out``
each read the partials and each wrote the canonical file, so the last writer
silently dropped the other's rows (and then deleted the partials it never
merged).  These tests reproduce that exact "two workers, one --out" shape and
assert the rows survive, that an incomplete merge fails loudly without touching
the file or the partials, and that the write is atomic and protocol-guarded.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from udwm.utils.atomic_merge import (
    merge_rows_additive,
    protocol_mismatch,
    read_row_map,
)

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "_probe" / "merge_tests"


def _row(seed, variant, marker=0.0):
    return {"seed": seed, "variant": variant, "value": marker,
            "teacher_final_checksum": 1.0 + seed}


def _write_partial(out, seed, variants):
    p = out.with_name("%s_seed%d.partial.json" % (out.stem, seed))
    p.write_text(json.dumps({"rows": [_row(seed, v) for v in variants]}), encoding="utf-8")
    return p


def _protocol(config="c.yaml", steps=1800, variants=("ordinary", "identified_eq")):
    return {"config": config, "steps": steps, "variants": list(variants)}


@pytest.fixture()
def tmpdir(request):
    d = TMP / request.node.name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _merge(out, added, expected, protocol, **kw):
    return merge_rows_additive(
        out, added, expected, protocol,
        protocol_keys=("config", "steps", "variants"),
        cleanup_paths_fn=kw.pop("cleanup_paths_fn", None),
        verbose=False, **kw)


def test_two_workers_one_out_union_instead_of_clobber(tmpdir):
    """The original race: worker 2 finishing last must not drop worker 1's seeds."""
    out = tmpdir / "study.json"

    # worker A finished seeds 0-1 and merged them
    rows_a = [_row(0, "ordinary"), _row(1, "ordinary")]
    assert _merge(out, rows_a, {(0, "ordinary"), (1, "ordinary")}, _protocol())

    # worker B finished seeds 2-3 and merges its own expected set (its rows only)
    rows_b = [_row(2, "ordinary"), _row(3, "ordinary")]
    assert _merge(out, rows_b, {(2, "ordinary"), (3, "ordinary")}, _protocol())

    keys = set(read_row_map(out))
    assert keys == {(0, "ordinary"), (1, "ordinary"),
                    (2, "ordinary"), (3, "ordinary")}
    assert len(json.loads(out.read_text())["rows"]) == 4


def test_duplicate_key_last_writer_wins(tmpdir):
    out = tmpdir / "study.json"
    _merge(out, [_row(0, "ordinary", 1.0)], {(0, "ordinary")}, _protocol())
    _merge(out, [_row(0, "ordinary", 9.0)], {(0, "ordinary")}, _protocol())
    assert read_row_map(out)[(0, "ordinary")]["value"] == 9.0


def test_missing_row_fails_loudly_and_touches_nothing(tmpdir, capsys):
    out = tmpdir / "study.json"
    partial = _write_partial(out, 0, ["ordinary", "identified_eq"])
    before = out.read_text() if out.exists() else None

    ok = _merge(out, [_row(0, "ordinary")],
                {(0, "ordinary"), (0, "identified_eq")}, _protocol())

    assert ok is False
    assert (out.read_text() if out.exists() else None) == before  # file untouched
    assert partial.exists()                                       # partial kept
    assert "FATAL" in capsys.readouterr().out


def test_protocol_mismatch_refuses_to_union(tmpdir):
    out = tmpdir / "study.json"
    _merge(out, [_row(0, "ordinary")], {(0, "ordinary")},
           _protocol(config="hopper.yaml"))
    snapshot = out.read_text()

    ok = _merge(out, [_row(1, "ordinary")], {(1, "ordinary")},
                _protocol(config="walker2d.yaml"))

    assert ok is False
    assert out.read_text() == snapshot                 # nothing merged
    assert protocol_mismatch(out, _protocol(config="walker2d.yaml"),
                             ("config", "steps", "variants")) is True


def test_write_is_atomic_and_leaves_no_tmp(tmpdir):
    out = tmpdir / "study.json"
    assert _merge(out, [_row(0, "ordinary")], {(0, "ordinary")}, _protocol())
    assert out.exists()
    assert not out.with_name(out.name + ".tmp").exists()
    json.loads(out.read_text())  # parses


def test_partials_removed_only_on_success(tmpdir):
    out = tmpdir / "study.json"
    p0 = _write_partial(out, 0, ["ordinary"])
    p1 = _write_partial(out, 1, ["ordinary"])
    expected = {(0, "ordinary"), (1, "ordinary")}
    cleanup = lambda: [p0, p1]

    # failure path: partials survive for resume
    assert _merge(out, [_row(0, "ordinary")], expected, _protocol(),
                  cleanup_paths_fn=cleanup) is False
    assert p0.exists() and p1.exists()

    # success path: partials removed
    assert _merge(out, [_row(0, "ordinary"), _row(1, "ordinary")], expected,
                  _protocol(), cleanup_paths_fn=cleanup) is True
    assert not p0.exists() and not p1.exists()


def test_keep_partials_retains_them(tmpdir):
    out = tmpdir / "study.json"
    p0 = _write_partial(out, 0, ["ordinary"])
    assert _merge(out, [_row(0, "ordinary")], {(0, "ordinary")}, _protocol(),
                  cleanup_paths_fn=lambda: [p0], keep_partials=True) is True
    assert p0.exists()
