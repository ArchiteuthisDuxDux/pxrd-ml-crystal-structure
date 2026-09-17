"""
Пре-расчёт space group для всех структур crystalDB в изолированных
субпроцессах (spglib иногда сегфолтит - крашер помечается и обходится).

Результат: data/mp_sg.csv (mp_id, sg, status)
  sg: 1..230, либо -1 (не определено), -2 (ошибка), -3 (краш spglib)
  status: ok | crash | error | undef

Запуск: python precompute_mp_sg.py [--workers 5] [--chunk 300]
Возобновляемо: существующие id в csv пропускаются.
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
IDS_FILE = PIPE / "data" / "mp_ids.txt"
OUT_CSV = PIPE / "data" / "mp_sg.csv"
WORKER = PIPE / "src" / "mp_sg_worker.py"
PYTHON = sys.executable


def ensure_id_list():
    if IDS_FILE.exists():
        return [l.strip() for l in IDS_FILE.read_text(encoding="utf-8").split() if l.strip()]
    from ase.db import connect

    db = connect(str(PIPE / "data" / "MP.db"))
    ids = [str(r.id) for r in db.select()]
    IDS_FILE.write_text("\n".join(ids), encoding="utf-8")
    return ids


def load_results():
    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV, dtype={"mp_id": str})
        return dict(zip(df["mp_id"], zip(df["sg"], df["status"])))
    return {}


def save_results(results, lock=None):
    if lock is not None:
        lock.acquire()
    try:
        rows = [(k, v[0], v[1]) for k, v in list(results.items())]
    finally:
        if lock is not None:
            lock.release()
    pd.DataFrame(rows, columns=["mp_id", "sg", "status"]).to_csv(OUT_CSV, index=False)


def process_segment(seg_ids, index_of, results, lock, chunk, timeout):
    """Обрабатывает сегмент позиций [0, len(seg_ids)) с субпроцессами."""
    pos = 0
    while pos < len(seg_ids):
        try:
            while pos < len(seg_ids) and seg_ids[pos] in results:
                pos += 1
            if pos >= len(seg_ids):
                break

            end = min(pos + chunk, len(seg_ids))
            idx_start = index_of[seg_ids[pos]]
            idx_end = index_of[seg_ids[end - 1]] + 1  # исключающая граница
            cmd = [PYTHON, str(WORKER), str(idx_start), str(idx_end)]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            )

            last_parsed = pos - 1
            deadline = time.time() + timeout
            try:
                for line in proc.stdout:
                    line = line.strip()
                    if not line or " " not in line:
                        if time.time() > deadline:
                            proc.kill()
                            break
                        continue
                    sid, sg_s = line.split()
                    sg_i = int(sg_s)
                    if sg_i >= 1:
                        status = "ok"
                    elif sg_i == -2:
                        status = "error"
                    else:
                        status = "undef"
                    with lock:
                        results[sid] = (sg_i, status)
                    last_parsed += 1
                    if time.time() > deadline:
                        proc.kill()
                        break
                proc.wait(timeout=30)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

            if proc.poll() not in (0, None):
                # краш: виновник - следующий после последнего успешного
                with lock:
                    if last_parsed + 1 < end:
                        results[seg_ids[last_parsed + 1]] = (-3, "crash")
                pos = last_parsed + 2
            else:
                pos = end
        except Exception as e:
            with lock:
                print(f"  [segment error] pos={pos}: {type(e).__name__}: {e}", flush=True)
            pos += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=300)
    args = ap.parse_args()

    ids = ensure_id_list()
    index_of = {sid: i for i, sid in enumerate(ids)}
    results = load_results()
    todo = [sid for sid in ids if sid not in results]
    print(f"Всего id: {len(ids)} | уже готово: {len(results)} | осталось: {len(todo)}", flush=True)
    if not todo:
        return

    lock = threading.Lock()
    n_par = args.workers
    segments = [todo[i::n_par] for i in range(n_par)]

    threads = []
    for seg in segments:
        if not seg:
            continue
        t = threading.Thread(
            target=process_segment,
            args=(seg, index_of, results, lock, args.chunk, args.chunk * 8 + 120),
        )
        t.start()
        threads.append(t)

    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(20)
        with lock:
            print(f"  прогресс: {len(results)}/{len(ids)} ({time.time()-t0:.0f} c)", flush=True)
        save_results(results, lock)
    for t in threads:
        t.join()

    save_results(results)
    statuses = pd.Series([v[1] for v in results.values()]).value_counts()
    print("Готово. Статусы:")
    print(statuses.to_string())


if __name__ == "__main__":
    main()
