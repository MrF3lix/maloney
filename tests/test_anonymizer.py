import io
import json
import threading
import time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from anonymizer.globi import GlobiClient, GlobiError
from anonymizer.pack import pack, pack_h5
from anonymizer.synth import ClipReader, QueueMonitor, generate, plan_clips, plan_speakers, read_results, speaker_table
from maloney.data.stt4sg import STT4SGDataModule, read_h5_index, read_split

REGIONS = ["Basel", "Bern", "Zürich"]
SR = 16_000


def make_meta(n_speakers_per_region=3, n_clips=4) -> pd.DataFrame:
    rows = []
    for r, region in enumerate(REGIONS):
        for s in range(n_speakers_per_region):
            cid = f"{region[:2]}{s}"
            for c in range(n_clips):
                rows.append({"path": f"{cid}/clip{c}.mp3", "duration": 1.0 + c, "sentence": f"Satz {cid} {c}.",
                             "sentence_source": "news", "client_id": cid, "dialect_region": region,
                             "canton": region[:2].upper(), "zipcode": str(4000 + 1000 * r + s),
                             "age": "thirties", "gender": "female" if s % 2 else "male"})
    return pd.DataFrame(rows)


def test_plan_speakers_picks_donors_from_other_regions():
    speakers = speaker_table(make_meta())
    plan = plan_speakers(speakers, speakers, n_donors=4, seed=0)
    assert len(plan) == 4 * len(speakers) and plan["client_id"].is_unique
    merged = plan.merge(speakers, left_on="donor_client_id", right_on="client_id", suffixes=("", "_donor"))
    assert (merged["dialect_region_donor"] != merged["dialect_region"]).all()
    # labels of the source, voice of the donor
    assert (merged["gender"] == merged["gender_donor"]).all()
    assert set(plan.loc[plan["source_client_id"] == "Ba0", "zipcode"]) == {"4000"}
    # both other regions are used before one is used twice
    assert plan.groupby("source_client_id")["donor_dialect_region"].nunique().eq(2).all()
    # a speaker's donors do not depend on the other speakers in the plan
    alone = plan_speakers(speakers[speakers["client_id"] == "Ba0"], speakers, n_donors=4, seed=0)
    assert alone["donor_client_id"].tolist() == plan.loc[plan["source_client_id"] == "Ba0", "donor_client_id"].tolist()


def test_plan_speakers_with_too_few_donors():
    speakers = speaker_table(make_meta(n_speakers_per_region=1))
    assert len(plan_speakers(speakers, speakers, n_donors=5, seed=0)) == 3 * 2


def test_plan_clips():
    meta = make_meta()
    speakers = plan_speakers(speaker_table(meta), speaker_table(meta), n_donors=2, seed=0)
    plan = plan_clips(meta, speakers, clips_per_speaker=2, max_target_seconds=3.5, seed=0)
    assert len(plan) == len(speakers) * 2 and plan["path"].is_unique
    assert (plan["duration"] <= 3.5).all()
    # every donor of a source re-speaks the same clips
    assert plan.groupby("source_client_id").apply(
        lambda g: g.groupby("client_id")["source_path"].apply(frozenset).nunique()).eq(1).all()
    row = plan.iloc[0]
    assert row["path"] == f"{row['client_id']}/{row['source_path'].split('/')[1].replace('.mp3', '.wav')}"
    all_clips = plan_clips(meta, speakers, clips_per_speaker=None, max_target_seconds=27, seed=0)
    assert len(all_clips) == len(speakers) * 4


def write_base_h5(path, meta, rng):
    audio = [rng.integers(-1000, 1000, int(d * SR)).astype(np.int16) for d in meta["duration"]]
    lengths = np.array([len(a) for a in audio])
    with h5py.File(path, "w") as f:
        f["audio"] = np.concatenate(audio)
        f.create_dataset("path", data=meta["path"].tolist(), dtype=h5py.string_dtype())
        f["offset"] = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        f["length"] = lengths.astype(np.int64)
        f.attrs["sample_rate"] = SR
    return audio


def test_pack_h5_links_the_base_audio(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    meta = make_meta(1, 2)
    (tmp_path / "base").mkdir()
    base_audio = write_base_h5(tmp_path / "base" / "audio_16k.h5", meta, rng)
    wav = tmp_path / "synth.wav"
    sf.write(wav, rng.uniform(-0.5, 0.5, 24_000).astype(np.float32), 24_000)  # 1 s at globi's rate
    (tmp_path / "synth").mkdir()
    out = tmp_path / "synth" / "audio_16k.h5"
    pack_h5(tmp_path / "base" / "audio_16k.h5", [("new/a.wav", wav)], out)

    monkeypatch.chdir("/")  # the link is relative to the file, not to the working directory
    index, sr = read_h5_index(out)
    assert sr == SR and index.index.tolist() == [*meta["path"], "new/a.wav"]
    with h5py.File(out) as f:
        audio = f["audio"]
        offset, length = index.loc[meta["path"].iloc[1]]
        np.testing.assert_array_equal(audio[offset: offset + length], base_audio[1])
        offset, length = index.loc["new/a.wav"]
        assert length == SR and np.abs(audio[offset: offset + length]).max() > 0

    (tmp_path / "base" / "audio_16k.h5").unlink()
    with pytest.raises(FileNotFoundError, match="links to"):
        read_h5_index(out)


class FakeGlobi(BaseHTTPRequestHandler):
    """The gateway's API; a job completes at once and returns the target at 24 kHz."""
    jobs: dict = {}
    counter = iter(range(10**9))
    fail_text = "kaputt"

    def log_message(self, *args):
        pass

    def reply(self, code, body=b"", content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, json.dumps({"status": "ok", "workers": 1, "queued": 0}).encode())
        job_id, *rest = self.path.removeprefix("/jobs/").split("/")
        job = self.jobs.get(job_id)
        if job is None:
            return self.reply(404, b'{"detail": "job not found"}')
        if rest:
            return self.reply(200, job["result"], "audio/wav")
        self.reply(200, json.dumps({k: v for k, v in job.items() if k != "result"}).encode())

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        message = BytesParser(policy=HTTP).parsebytes(
            f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + body)
        parts = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                 for p in message.iter_parts()}
        assert parts["language"].decode().startswith("ch_") and sf.info(io.BytesIO(parts["reference_wav"])).duration > 0
        target, sr = sf.read(io.BytesIO(parts["target_wav"]), dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, np.interp(np.arange(len(target) * 3 // 2) / 1.5, np.arange(len(target)), target),
                 24_000, format="WAV")
        job_id = f"job{next(self.counter)}"
        text = parts["text"].decode()
        failed = self.fail_text in text
        self.jobs[job_id] = {"id": job_id, "status": "failed" if failed else "completed", "transcript": text,
                             "error": "boom" if failed else "", "result": buf.getvalue()}
        self.reply(202, json.dumps({"id": job_id, "status": "queued"}).encode())

    def do_DELETE(self):
        self.jobs.pop(self.path.removeprefix("/jobs/"), None)
        self.reply(204)


@pytest.fixture
def globi():
    FakeGlobi.jobs = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGlobi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield GlobiClient(f"http://127.0.0.1:{server.server_port}", retries=0, poll_interval=0.01)
    server.shutdown()


def test_globi_client(globi):
    target = io.BytesIO()
    sf.write(target, np.zeros(SR, dtype=np.float32), SR, format="WAV")
    wav, job = globi.anonymize(target.getvalue(), target.getvalue(), "ch_zh", "Hallo.")
    assert sf.info(io.BytesIO(wav)).samplerate == 24_000 and job["transcript"] == "Hallo."
    assert FakeGlobi.jobs == {}  # deleted after fetching
    with pytest.raises(GlobiError, match="boom"):
        globi.anonymize(target.getvalue(), target.getvalue(), "ch_zh", "kaputt")


def test_generate_and_pack_is_a_drop_in_test_split(tmp_path, globi):
    """End to end on a fake gateway: the output folder loads with the data module like the source."""
    rng = np.random.default_rng(0)
    src = tmp_path / "src"
    meta = make_meta()
    split_dir = src / "splits" / "speakers_v1"
    split_dir.mkdir(parents=True)
    for split, region_speakers in [("train", ["0"]), ("valid", ["1"]), ("test", ["2"])]:
        meta[meta["client_id"].str[-1].isin(region_speakers)].to_csv(split_dir / f"{split}.tsv", sep="\t",
                                                                      index=False, quoting=3)
    write_base_h5(src / "audio_16k.h5", meta, rng)
    data_cfg = SimpleNamespace(data_dir=str(src), split_dir=str(split_dir), train_split="train_all",
                               audio_h5=str(src / "audio_16k.h5"), geo_table=None)

    test = read_split(src, "test", split_dir=split_dir)
    speakers = plan_speakers(speaker_table(test), speaker_table(test), n_donors=2, seed=0)
    plan = plan_clips(test, speakers, clips_per_speaker=2, max_target_seconds=27, seed=0)
    plan.loc[0, "sentence"] = "kaputt"  # one job fails on the gateway
    out = tmp_path / "out"
    reader = ClipReader(src / "audio_16k.h5")
    generate(plan, test, reader, globi, out, reference_seconds=3, concurrency=4, seed=0)
    results = read_results(out / "results.jsonl")
    assert sum(r["status"] == "completed" for r in results.values()) == len(plan) - 1

    plan.loc[0, "sentence"] = "repariert"  # a rerun only retries the failed clip
    n_jobs = len(FakeGlobi.jobs)
    generate(plan, test, reader, globi, out, reference_seconds=3, concurrency=4, seed=0)
    assert len(read_results(out / "results.jsonl")) == len(plan)

    pack(data_cfg, plan, read_results(out / "results.jsonl"), out, "test", include_original=True)
    dm = STT4SGDataModule(str(out), str(out / "audio_16k.h5"), split_dir=str(out / "splits" / "speakers_v1"))
    dm.setup()
    assert len(dm.datasets["train"]) == len(read_split(src, "train", split_dir=split_dir))
    test_meta = dm.datasets["test"].meta
    assert len(test_meta) == len(test) + len(plan) and test_meta["synthetic"].sum() == len(plan)
    synth_item = dm.datasets["test"][int(np.flatnonzero(test_meta["synthetic"])[0])]
    source_row = test_meta[test_meta["path"] == test_meta.loc[test_meta["synthetic"], "source_path"].iloc[0]].iloc[0]
    assert synth_item["zipcode"] == source_row["zipcode"] and synth_item["client_id"] != source_row["client_id"]
    assert len(synth_item["audio"]) == pytest.approx(source_row["length"], abs=2)
    assert len(pd.read_csv(out / "speakers.csv")) == len(speakers)


def test_queue_monitor_logs_the_queue(globi, caplog):
    caplog.set_level("INFO", logger="anonymizer.synth")
    with QueueMonitor(globi, interval=0.05, progress=lambda: "this run: 1/2 done"):
        time.sleep(0.3)
    lines = [r.getMessage() for r in caplog.records if "globi queue" in r.getMessage()]
    assert len(lines) >= 2 and lines[0] == "globi queue: 0 queued, 1 workers | this run: 1/2 done"
    caplog.clear()
    with QueueMonitor(GlobiClient("http://127.0.0.1:9", retries=0, timeout=0.1), interval=0.05):
        time.sleep(0.2)
    assert any("not reachable" in r.getMessage() for r in caplog.records)
    with QueueMonitor(globi, interval=None):  # off
        pass
