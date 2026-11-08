# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Measures the running service: latency percentiles and throughput.

    python scripts/loadtest.py --url http://localhost:8000 --requests 300 --concurrency 4

Sends the same 600 x 450 JPEG (the size of a HAM10000 image) over and over,
after a short warm-up, and reports p50 / p95 / p99 latency and requests per
second. Prints Markdown, so CI can put it in the job summary.
"""
import argparse
import asyncio
import io
import statistics
import time

import httpx
import numpy as np
from PIL import Image


def sample_jpeg() -> bytes:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:450, 0:600]
    blob = np.exp(-(((xx - 300) / 120) ** 2 + ((yy - 225) / 90) ** 2))   # a dark round "lesion" on skin tone
    base = np.stack([200 - 110 * blob, 150 - 90 * blob, 130 - 80 * blob], -1) + rng.normal(0, 6, (450, 600, 3))
    buf = io.BytesIO()
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


async def main(url, n, concurrency):
    data = sample_jpeg()
    async with httpx.AsyncClient(base_url=url, timeout=30) as client:
        for _ in range(10):   # warm-up
            (await client.post("/v1/predict", files={"file": ("x.jpg", data, "image/jpeg")})).raise_for_status()
        sem = asyncio.Semaphore(concurrency)
        lat, errors = [], 0

        async def one():
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                r = await client.post("/v1/predict", files={"file": ("x.jpg", data, "image/jpeg")})
                lat.append((time.perf_counter() - t0) * 1000)
                errors += r.status_code != 200

        t0 = time.perf_counter()
        await asyncio.gather(*[one() for _ in range(n)])
        wall = time.perf_counter() - t0
    q = statistics.quantiles(lat, n=100)
    print(f"| requests | concurrency | p50 ms | p95 ms | p99 ms | req/s | errors |\n|---|---|---|---|---|---|---|\n"
          f"| {n} | {concurrency} | {q[49]:.0f} | {q[94]:.0f} | {q[98]:.0f} | {n / wall:.1f} | {errors} |")
    return errors


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--concurrency", type=int, default=4)
    a = p.parse_args()
    raise SystemExit(1 if asyncio.run(main(a.url, a.requests, a.concurrency)) else 0)
