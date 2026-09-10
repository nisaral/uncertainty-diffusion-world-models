# -*- coding: utf-8 -*-
"""Race-safe additive merge for per-(seed, variant) partial payloads.

Shared by every multi-worker runner in the repo: the policy 2x2 split driver
(``udwm/scripts/run_policy_2x2_split_seeds.py``) and both probes
(``probe_crn_bias.py``, ``probe_gradient_interference.py``).

The failure mode this exists to prevent: two processes that ran the same study
into one ``--out`` each read the partials, each write the canonical file, and the
last writer silently drops the other's rows (and, without ``--keep-partials``,
then deletes the partials it never merged).  The fix is to make the merge a
*serialized read-modify-write of the canonical file* rather than an overwrite.

Guarantees:
  * serialized - an advisory exclusive lock on ``<out>.lock`` (``fcntl`` on
    POSIX, ``msvcrt`` on Windows, no-op only if neither exists) covers the whole
    read-modify-write, so concurrent finishers union instead of clobbering;
  * additive - rows already in ``out`` are read back and unioned; the calling
    invocation wins on duplicate ``(seed, variant)`` keys;
  * complete - every requested ``(seed, variant)`` must be present, else the
    merge fails loudly and leaves both the canonical file and the partials
    untouched so the run can resume;
  * atomic - the payload goes to ``<out>.tmp`` and is ``os.replace``d, so a
    crash can never leave a truncated canonical file that still looks parseable;
  * protocol-guarded - rows written under a different protocol
    (config / steps / arm set) are never silently unioned into this one.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "read_row_map",
    "protocol_mismatch",
    "exclusive_lock",
    "merge_rows_additive",
]


def read_row_map(out) -> dict:
    """Existing canonical rows as ``{(seed, variant): row}``; ``{}`` if absent/broken."""
    out = Path(out)
    if not out.exists():
        return {}
    try:
        payload = json.loads(out.read_text(encoding="utf-8"))
        return {(int(r["seed"]), str(r["variant"])): r for r in payload.get("rows", [])}
    except (ValueError, KeyError, TypeError, OSError):
        return {}


def protocol_mismatch(out, protocol, keys) -> bool:
    """True when ``out`` already holds rows from a different protocol."""
    try:
        old = json.loads(Path(out).read_text(encoding="utf-8")).get("protocol", {})
    except (ValueError, OSError):
        return True
    return any(old.get(k) != protocol.get(k) for k in keys)


@contextmanager
def exclusive_lock(out):
    """Advisory exclusive lock on ``<out>.lock``; no-op where unsupported."""
    handle = None
    mode = None
    path = str(Path(out)) + ".lock"
    try:
        import fcntl  # POSIX
        handle = open(path, "a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        mode = "fcntl"
    except Exception:
        if handle is not None:
            handle.close()
        handle = None
        try:
            import msvcrt  # Windows
            handle = open(path, "a+")
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            mode = "msvcrt"
        except Exception:
            if handle is not None:
                handle.close()
            handle = None
    try:
        yield handle
    finally:
        if handle is not None:
            try:
                if mode == "fcntl":
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                elif mode == "msvcrt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            handle.close()


def _sorted_keys(merged, variant_order=None):
    if variant_order:
        order = {v: i for i, v in enumerate(variant_order)}
        width = len(order)
        return sorted(merged, key=lambda k: (k[0], order.get(k[1], width), k[1]))
    return sorted(merged)


def merge_rows_additive(out, added_rows, expected, protocol, *,
                        protocol_keys=("config", "steps", "variants"),
                        pairing_fn=None, extras_fn=None,
                        cleanup_paths_fn=None, variant_order=None,
                        keep_partials=False, delete_partials=True,
                        label="merge", verbose=True):
    """Union ``added_rows`` into the canonical ``out`` under an exclusive lock.

    ``expected`` is the set of ``(seed, variant)`` pairs that must all be present
    afterwards.  Returns ``True`` on a complete, written merge and ``False`` on a
    loud failure (protocol mismatch, or rows missing) with the file and partials
    left untouched.  ``pairing_fn(rows)`` / ``extras_fn(rows)`` add the
    runner-specific payload blocks (teacher pairing, summary, ...).
    """
    out = Path(out)
    expected = {(int(s), str(v)) for s, v in expected}
    added = {(int(r["seed"]), str(r["variant"])): r for r in added_rows}

    def say(msg, force=False):
        # FATAL diagnostics are never gated on verbose: a refused merge
        # must be visible in every caller's log, including a quiet one.
        if verbose or force:
            print("[{}] {}".format(label, msg))

    with exclusive_lock(out):
        merged = read_row_map(out)
        if merged and protocol_mismatch(out, protocol, protocol_keys):
            say("FATAL: existing rows in {} are from a different protocol "
                "({}); refusing to union. Use a fresh --out for a different "
                "study/budget.".format(out.name, "/".join(protocol_keys)), force=True)
            return False

        merged.update(added)
        rows = [merged[k] for k in _sorted_keys(merged, variant_order)]
        missing = sorted(expected - set(merged))
        if missing:
            shown = ", ".join("s{}:{}".format(s, v) for s, v in missing[:8])
            more = " ..." if len(missing) > 8 else ""
            say("FATAL: expected {} (seed, variant) rows for the requested "
                "seeds x arms, found {}; {} missing: {}{}".format(
                    len(expected), len(rows), len(missing), shown, more), force=True)
            return False

        payload = {"protocol": protocol, "rows": rows}
        if pairing_fn is not None:
            payload["teacher_pairing"] = pairing_fn(rows)
        if extras_fn is not None:
            payload.update(extras_fn(rows))

        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, out)
        say("merged {} rows -> {} (expected {} for the requested seeds x arms)"
            .format(len(rows), out, len(expected)))

        if delete_partials and not keep_partials and cleanup_paths_fn is not None:
            for p in cleanup_paths_fn():
                if p is not None:
                    Path(p).unlink(missing_ok=True)
            say("removed partial files (--keep-partials to retain)")
        return True
