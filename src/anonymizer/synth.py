"""Plan and generate synthetic speakers: the words and dialect of a source speaker, the voice of a donor.

For every source speaker X (e.g. every test speaker), `n_donors` donors are picked from other dialect
regions (round robin over the regions, so they spread out), and every donor re-speaks the same clips
of X with globi in X's dialect. Each (X, donor) pair is a new synthetic speaker with its own
`client_id`: located where X lives (dialect_region, canton, zipcode of X), with the age and gender of
the donor (the voice). The dialect can then only come from the dialect token and X's words, not from
the voice.

The plan is deterministic: a speaker's donors and clips depend only on the seed and its client_id,
so a rerun with the same settings resumes where the last one stopped.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import pandas as pd
import soundfile as sf

from anonymizer.globi import GlobiClient, GlobiError
from maloney.data.meta import read_h5_index

log = logging.getLogger(__name__)

# dialect_region of STT4SG-350 -> globi language token
LANGUAGES = {"Basel": "ch_bs", "Bern": "ch_be", "Graubünden": "ch_gr", "Innerschweiz": "ch_in",
             "Ostschweiz": "ch_os", "Wallis": "ch_vs", "Zürich": "ch_zh"}
# Namespace of the synthetic client_ids and of the default output name.
NAMESPACE = "stt4sg-350_synth"


def speaker_rng(seed: int, client_id: str) -> np.random.Generator:
    """Random draws of one speaker, independent of which other speakers are in the plan."""
    return np.random.default_rng([seed, zlib.crc32(client_id.encode())])


def speaker_table(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per speaker of a split's clip metadata."""
    mode = lambda s: s.mode().iloc[0]  # noqa: E731
    return (meta.groupby("client_id")
            .agg(dialect_region=("dialect_region", mode), canton=("canton", mode), zipcode=("zipcode", mode),
                 age=("age", mode), gender=("gender", mode), n_clips=("path", "size"),
                 hours=("duration", lambda d: d.sum() / 3600))
            .reset_index())


def plan_speakers(sources: pd.DataFrame, donors: pd.DataFrame, n_donors: int, seed: int) -> pd.DataFrame:
    """The synthetic speakers: `n_donors` donors from other dialect regions for every source speaker.

    `sources` and `donors` are speaker tables (speaker_table). The regions are visited in a random
    order per source, one donor each, then again until `n_donors` are picked.
    """
    if unknown := set(sources["dialect_region"]) - set(LANGUAGES):
        raise ValueError(f"no globi language for the dialect regions {sorted(unknown)}")
    rows = []
    for x in sources.sort_values("client_id").itertuples():
        rng = speaker_rng(seed, x.client_id)
        pool = donors[(donors["dialect_region"] != x.dialect_region) & (donors["client_id"] != x.client_id)]
        queues = {region: list(rng.permutation(group["client_id"].sort_values().to_numpy()))
                  for region, group in pool.groupby("dialect_region")}
        picked = []
        for region in rng.permutation(sorted(queues)).tolist() * n_donors:
            if len(picked) == n_donors:
                break
            if queues[region]:
                picked.append(queues[region].pop(0))
        if len(picked) < n_donors:
            log.warning("%s: only %d donors from other dialect regions", x.client_id, len(picked))
        for donor_id in picked:
            donor = pool[pool["client_id"] == donor_id].iloc[0]
            rows.append({
                "client_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{NAMESPACE}/{x.client_id}/{donor_id}")),
                "source_client_id": x.client_id, "donor_client_id": donor_id,
                "dialect_region": x.dialect_region, "canton": x.canton, "zipcode": x.zipcode,
                "age": donor["age"], "gender": donor["gender"],
                "donor_dialect_region": donor["dialect_region"], "donor_canton": donor["canton"],
                "donor_zipcode": donor["zipcode"], "language": LANGUAGES[x.dialect_region],
            })
    return pd.DataFrame(rows)


def plan_clips(meta: pd.DataFrame, speakers: pd.DataFrame, clips_per_speaker: int | None,
               max_target_seconds: float, seed: int) -> pd.DataFrame:
    """One row per synthetic clip: every donor of a source speaker re-speaks the same clips of it.

    `clips_per_speaker` clips are drawn per source speaker (None: all of them); clips longer than
    `max_target_seconds` (globi's limit) are skipped. The columns are those of the release tsv plus
    the provenance of speakers (plan_speakers) and `source_path`; `path` is
    <synthetic client_id>/<file name of the source clip>.wav.
    """
    too_long = meta["duration"] > max_target_seconds
    if too_long.any():
        log.warning("%d clips are longer than %g s and are not re-spoken", too_long.sum(), max_target_seconds)
    meta = meta[~too_long & meta["client_id"].isin(speakers["source_client_id"])]
    picked = []
    for client_id, clips in meta.groupby("client_id", sort=True):
        clips = clips.sort_values("path")
        if clips_per_speaker is not None and len(clips) > clips_per_speaker:
            rng = speaker_rng(seed, client_id)
            clips = clips.iloc[np.sort(rng.choice(len(clips), clips_per_speaker, replace=False))]
        picked.append(clips)
    clips = pd.concat(picked)[["path", "duration", "sentence", "sentence_source", "client_id"]]
    clips = clips.rename(columns={"path": "source_path", "client_id": "source_client_id"})
    plan = speakers.merge(clips, on="source_client_id")
    plan.insert(0, "path", [f"{cid}/{PurePosixPath(p).stem}.wav" for cid, p in zip(plan["client_id"], plan["source_path"])])
    return plan.sort_values(["source_client_id", "client_id", "source_path"]).reset_index(drop=True)


class ClipReader:
    """Clips of the packed audio file (audio_16k.h5) as 16-bit wav bytes."""

    def __init__(self, audio_h5: str | Path):
        self.audio_h5 = Path(audio_h5)
        self.index, self.sample_rate = read_h5_index(self.audio_h5)
        self._audio = h5py.File(self.audio_h5, "r")["audio"]
        self._lock = threading.Lock()

    def __contains__(self, path: str) -> bool:
        return path in self.index.index

    def audio(self, path: str) -> np.ndarray:
        offset, length = self.index.loc[path, ["offset", "length"]]
        with self._lock:
            return self._audio[offset: offset + length]

    def wav(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def reference(self, paths: list[str], seconds: float, gap: float = 0.2) -> bytes:
        """`paths` joined with `gap` s of silence until they last `seconds` (at most 55 s, globi takes 60)."""
        if not paths:
            raise ValueError(f"no reference clips: none of the donor's clips are in {self.audio_h5}")
        parts, total = [], 0
        silence = np.zeros(round(gap * self.sample_rate), dtype=np.int16)
        for path in paths:
            parts += [self.audio(path), silence]
            total += len(parts[-2])
            if total >= seconds * self.sample_rate:
                break
        return self.wav(np.concatenate(parts[:-1])[: 55 * self.sample_rate])


def read_results(results_file: Path) -> dict[str, dict]:
    """path -> the last record of that clip in results.jsonl."""
    if not results_file.is_file():
        return {}
    records = {}
    for line in results_file.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["path"]] = record
    return records


def completed_paths(out_dir: Path) -> set[str]:
    """Clips that were generated and whose wav is in out_dir/clips."""
    return {path for path, record in read_results(out_dir / "results.jsonl").items()
            if record["status"] == "completed" and (out_dir / "clips" / path).is_file()}


class QueueMonitor:
    """Logs the gateway's queue (/health) every `interval` seconds while the run is going.

    `progress` returns this run's own state as text (done, failed, in flight). An unreachable
    gateway is logged too, so a stalled run shows why.
    """

    def __init__(self, client: GlobiClient, interval: float, progress=lambda: ""):
        self.client, self.interval, self.progress = client, interval, progress
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="globi-queue-monitor", daemon=True)

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                health = self.client.health()
                log.info("globi queue: %s queued, %s workers | %s", health.get("queued"), health.get("workers"),
                         self.progress())
            except Exception as e:  # noqa: BLE001 - only logging; the jobs report their own errors
                log.warning("globi queue: gateway not reachable (%s) | %s", e, self.progress())

    def __enter__(self):
        if self.interval:
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()


def generate(plan: pd.DataFrame, donor_meta: pd.DataFrame, reader: ClipReader, client: GlobiClient,
             out_dir: Path, reference_seconds: float, concurrency: int, seed: int,
             max_consecutive_failures: int = 50, status_interval: float | None = 1.0):
    """Run every clip of the plan through globi that is not done yet.

    Writes out_dir/clips/<path> (globi's output as is) and appends one record per clip to
    out_dir/results.jsonl (status, job id, transcript, error), so a killed run resumes and failed
    clips are retried by the next run. Stops if `max_consecutive_failures` clips fail in a row
    (gateway down, wrong url). Every `status_interval` seconds the gateway's queue is logged
    (QueueMonitor; None or 0 turns it off).
    """
    clips_dir, results_file = out_dir / "clips", out_dir / "results.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = completed_paths(out_dir)
    todo = plan[~plan["path"].isin(done)]
    missing = ~todo["source_path"].map(reader.__contains__)
    if missing.any():
        log.warning("%d source clips are not in %s and are skipped", missing.sum(), reader.audio_h5)
        todo = todo[~missing]
    log.info("%d/%d clips already done, %d to generate with %s (%s)", len(done & set(plan["path"])), len(plan),
             len(todo), client, client.health())
    if todo.empty:
        return

    references = {}
    for donor_id in sorted(todo["donor_client_id"].unique()):
        paths = donor_meta.loc[donor_meta["client_id"] == donor_id, "path"]
        paths = [p for p in speaker_rng(seed, donor_id).permutation(np.sort(paths.to_numpy())) if p in reader]
        references[donor_id] = reader.reference(paths, reference_seconds)

    def run(row) -> dict:
        target = reader.wav(reader.audio(row.source_path))
        wav, job = client.anonymize(target, references[row.donor_client_id], row.language, row.sentence)
        out = clips_dir / row.path
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_bytes(wav)
        os.replace(tmp, out)
        return {"path": row.path, "status": "completed", "job_id": job["id"],
                "transcript": job.get("transcript", ""), "duration": float(job.get("duration") or 0)}

    start, n_done, n_failed, in_a_row = time.time(), 0, 0, 0

    def progress() -> str:
        return f"this run: {n_done}/{len(todo)} done, {n_failed} failed, {min(concurrency, len(todo) - n_done)} in flight"

    with (ThreadPoolExecutor(concurrency) as pool, results_file.open("a") as results,
          QueueMonitor(client, status_interval, progress)):
        futures = {pool.submit(run, row): row for row in todo.itertuples()}
        try:
            for future in as_completed(futures):
                row = futures[future]
                try:
                    record = future.result()
                    in_a_row = 0
                except Exception as e:  # noqa: BLE001 - recorded and retried by the next run
                    record = {"path": row.path, "status": "failed", "error": f"{type(e).__name__}: {e}"}
                    n_failed += 1
                    in_a_row += 1
                    log.warning("%s (source %s): %s", row.path, row.source_path, record["error"])
                    if in_a_row >= max_consecutive_failures:
                        raise GlobiError(f"{in_a_row} clips failed in a row, stopping; last error: {e}") from e
                results.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.flush()
                n_done += 1
                if n_done % 100 == 0 or n_done == len(todo):
                    rate = n_done / (time.time() - start)
                    log.info("%d/%d clips (%d failed), %.1f clips/s, eta %.0f min", n_done, len(todo), n_failed,
                             rate, (len(todo) - n_done) / rate / 60)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if n_failed:
        log.warning("%d clips failed (see results.jsonl); run again to retry them", n_failed)
