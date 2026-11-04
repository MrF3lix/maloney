"""Client of the globi gateway (/Users/felix/PhD/05_Tools/globi): one job re-speaks a target
recording in the voice of a reference recording, in a given dialect.

    POST /jobs                 multipart target_wav, reference_wav, language, text -> 202 {id, status}
    GET  /jobs/{id}            status queued | running | completed | failed, transcript, error
    GET  /jobs/{id}/result.wav
    DELETE /jobs/{id}
    GET  /health               {workers, queued}

Only the standard library is used, so the client runs in the training environment as is.
"""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.error
import urllib.request
import uuid

log = logging.getLogger(__name__)

# Worth retrying: the queue is full (429) or the gateway is restarting (5xx).
RETRY_STATUS = {429, 500, 502, 503, 504}


class GlobiError(RuntimeError):
    """A job was refused or failed; retrying the same request will not help."""


class GlobiClient:
    def __init__(self, url: str, timeout: float = 60.0, retries: int = 5, poll_interval: float = 2.0,
                 job_timeout: float = 900.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout

    def __repr__(self):
        return f"GlobiClient({self.url})"

    def _request(self, method: str, path: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
        """The response body; connection errors and RETRY_STATUS are retried with exponential backoff."""
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(f"{self.url}{path}", data=data, method=method, headers=headers or {})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code not in RETRY_STATUS or attempt == self.retries:
                    raise GlobiError(f"{method} {path}: {e.code} {body}") from None
                wait = float(e.headers.get("Retry-After") or 2 ** attempt)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise
                wait = 2 ** attempt
                log.debug("%s %s: %s, retrying in %.0f s", method, path, e, wait)
            time.sleep(wait)
        raise AssertionError("unreachable")

    def health(self) -> dict:
        return json.loads(self._request("GET", "/health"))

    def submit(self, target_wav: bytes, reference_wav: bytes, language: str, text: str | None = None) -> str:
        """Queue a job; returns its id. `text` is the Standard German transcript of the target."""
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        fields = [("language", language)] + ([("text", text)] if text else [])
        for name, value in fields:
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        for name, data in [("target_wav", target_wav), ("reference_wav", reference_wav)]:
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}.wav"\r\n'
                       "Content-Type: audio/wav\r\n\r\n".encode() + data + b"\r\n")
        body.write(f"--{boundary}--\r\n".encode())
        response = self._request("POST", "/jobs", body.getvalue(),
                                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return json.loads(response)["id"]

    def status(self, job_id: str) -> dict:
        return json.loads(self._request("GET", f"/jobs/{job_id}"))

    def result(self, job_id: str) -> bytes:
        return self._request("GET", f"/jobs/{job_id}/result.wav")

    def delete(self, job_id: str):
        self._request("DELETE", f"/jobs/{job_id}")

    def wait(self, job_id: str) -> dict:
        """Poll until the job is completed; raises GlobiError if it failed or took longer than job_timeout."""
        start = time.time()
        while True:
            job = self.status(job_id)
            if job["status"] == "completed":
                return job
            if job["status"] == "failed":
                raise GlobiError(f"job {job_id} failed: {job.get('error')}")
            if time.time() - start > self.job_timeout:
                raise GlobiError(f"job {job_id} still {job['status']} after {self.job_timeout:.0f} s")
            time.sleep(self.poll_interval)

    def anonymize(self, target_wav: bytes, reference_wav: bytes, language: str,
                  text: str | None = None) -> tuple[bytes, dict]:
        """Submit, wait and fetch one job: (result wav, job). The job is deleted on the gateway afterwards."""
        job_id = self.submit(target_wav, reference_wav, language, text)
        try:
            job = self.wait(job_id)
            return self.result(job_id), job
        finally:
            try:
                self.delete(job_id)
            except Exception as e:  # noqa: BLE001 - it expires on its own after the result TTL
                log.debug("could not delete job %s: %s", job_id, e)
