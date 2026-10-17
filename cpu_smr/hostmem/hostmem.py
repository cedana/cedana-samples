#!/usr/bin/env python3
"""
Host memory of every kind a checkpoint must get exactly right, shared by several processes.

Each region is one kind of mapping:

  anon          private anonymous memory (MAP_PRIVATE|MAP_ANONYMOUS), copy-on-write after fork
  shared-anon   MAP_SHARED|MAP_ANONYMOUS, inherited across fork
  shm           a named POSIX shared memory file in /dev/shm, kept
  ghost-shm     the same, unlinked once mapped (a ghost file for CRIU), as KV-cache offload
                (SGLang HiCache) shares a /dev/shm file between processes
  memfd         an anonymous file from memfd_create(2)
  file          a regular file in --dir, mapped MAP_SHARED, kept
  ghost-file    the same, unlinked once mapped
  sparse        a regular file in --dir with data at a few offsets only (past 4 GiB if it is
                that big) and holes everywhere else, kept
  ghost-sparse  the same, unlinked once mapped

Page 0 of every region is its comm page; the rest has a layout of data pages (unique to the
region and the offset), pages written as zeros, holes (never written), pages that are zero
but for their last bytes (head-zero) or first bytes (tail-zero), and a partly written last
page. A dense layout repeats that mix over the whole region; a sparse one places it at a few
offsets: the start, 1 MiB, either side of 2 GiB and 4 GiB, and the end.

--procs children are forked. Each maps the file-backed regions itself (its own mapping of
the inherited fd), then the parent unlinks the ghosts and closes every fd, so only mappings
keep them, and prints READY. Every process keeps verifying every region. Each child, after
each pass, writes its pass count into its slot of each shared region's comm page, and into
its private copy of each anon region's. The parent logs `OK <c0> <c1> ...`, the lowest count
it sees for each child across shared regions: they keep advancing only if the mappings are
still shared (not private copies). Its own anon comm pages must stay zero: a child's write
there must not be visible. Any error is logged as `MISMATCH ...` and the process exits 1.

With --gate PATH, once the file exists the parent verifies everything once more, requires
every child's counter to advance again, stops the children, prints PASS and exits 0, so a
test can checkpoint and restore it mid-run, then create the gate. Without it, it runs until
killed. Stdlib only, no root needed.

Usage: hostmem.py [--regions KIND:SIZE[:dense|sparse],...] [--preset NAME] [--procs N]
                  [--dir DIR] [--gate PATH] [--log PATH]
"""

import argparse
import mmap
import os
import signal
import struct
import sys
import time

PAGE = mmap.PAGESIZE
EDGE = 16  # the non-zero bytes of a head-zero or tail-zero page
ZERO = bytes(PAGE)
SHM_DIR = "/dev/shm"
SHM_RESERVE = 8 << 20  # left free in /dev/shm for semaphores and the like

# The mix of a dense layout, repeated page by page
CYCLE = ["data", "zero", "hole", "data", "headzero", "zero", "tailzero", "hole"]

PRESETS = {
    "small": "anon:1M,shared-anon:1M,shm:1M,ghost-shm:2M,memfd:1M,file:1M,ghost-file:1M",
    "large": "anon:256M,shared-anon:256M,shm:256M,ghost-shm:1G,memfd:256M,ghost-file:256M,"
             "ghost-sparse:5G",
    "sparse": "sparse:5G,ghost-sparse:5G",
    "hicache": "ghost-shm:1G",
}

out = sys.stdout


def log(msg):
    out.write(msg + "\n")
    out.flush()


def fail(msg):
    log(f"MISMATCH {msg}")
    sys.exit(1)


def parse_size(s):
    units = {"K": 10, "M": 20, "G": 30, "T": 40}
    if s[-1].upper() in units:
        return int(float(s[:-1]) * (1 << units[s[-1].upper()]))
    return int(s)


def human(n):
    for unit, shift in (("G", 30), ("M", 20), ("K", 10)):
        if n >= 1 << shift:
            return f"{n / (1 << shift):.4g}{unit}"
    return str(n)


class Region:
    """A mapping with a page layout to fill and verify, shared across fork or not"""

    shared = True  # do the children's writes reach the parent?

    def __init__(self, index, kind, size, dense):
        # A page-aligned size loses 1000 bytes, so the last page is always partly written
        self.index, self.kind, self.dense = index, kind, dense
        self.size = size - 1000 if size % PAGE == 0 else size
        self.tail = self.size - self.size % PAGE
        self.mem = None
        self.fd = -1
        self.path = None

    def describe(self):
        layout = "dense" if self.dense else "sparse"
        where = f" at {self.path}" if self.path else ""
        return f"{self.kind}[{self.index}] {human(self.size)} {layout}{where}"

    def allocate(self):
        raise NotImplementedError

    def remap(self):
        """In a child: replace the inherited mapping with one of its own, if it can"""
        if self.fd >= 0:
            old, self.mem = self.mem, self._map(self.fd)
            old.close()

    def release(self, unlink=True):
        """Once every process has mapped it: unlink a ghost, and close the fd"""
        if unlink and self.path and self.kind.startswith("ghost-"):
            os.unlink(self.path)
            self.path += " (unlinked)"
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def _map(self, fd):
        return mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def _open(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(self.fd, self.size)
        self.mem = self._map(self.fd)

    # The layout

    def data(self, off):
        """A data page: unique to this region and this offset"""
        return struct.pack("<QQ", off, self.index + 1) * (PAGE // 16)

    def expected(self, off, kind):
        if kind == "data":
            return self.data(off)
        if kind == "headzero":
            return ZERO[:PAGE - EDGE] + self.data(off)[:EDGE]
        if kind == "tailzero":
            return self.data(off)[:EDGE] + ZERO[EDGE:]
        return ZERO  # zero and hole

    def pages(self):
        """(offset, kind) of every page checked, past the comm page and before the last"""
        if self.dense:
            for off in range(PAGE, self.tail, PAGE):
                yield off, CYCLE[off // PAGE % len(CYCLE)]
            return
        anchors = [PAGE, 1 << 20, (2 << 30) - 4 * PAGE, (4 << 30) - 4 * PAGE,
                   self.tail - len(CYCLE) * PAGE]
        seen = set()
        for anchor in sorted(a for a in anchors if a >= PAGE):
            for i, kind in enumerate(CYCLE):
                off = anchor + i * PAGE
                if off >= self.tail or off in seen:
                    continue
                seen.add(off)
                yield off, kind

    def fill(self):
        for off, kind in self.pages():
            if kind != "hole":
                self.mem[off:off + PAGE] = self.expected(off, kind)
        self.mem[self.tail:self.size] = self.data(self.tail)[:self.size - self.tail]

    def verify(self, who):
        for off, kind in self.pages():
            if self.mem[off:off + PAGE] != self.expected(off, kind):
                fail(f"{who}: {self.describe()}: {kind} page at {off}")
        if self.mem[self.tail:self.size] != self.data(self.tail)[:self.size - self.tail]:
            fail(f"{who}: {self.describe()}: partly written last page at {self.tail}")

    # The comm page: one 8-byte counter per child

    def counter(self, k):
        return int.from_bytes(self.mem[8 * k:8 * k + 8], "little")

    def set_counter(self, k, n):
        self.mem[8 * k:8 * k + 8] = n.to_bytes(8, "little")


class AnonRegion(Region):
    def __init__(self, index, kind, size, dense):
        super().__init__(index, kind, size, dense)
        self.shared = kind == "shared-anon"

    def allocate(self):
        flags = mmap.MAP_SHARED if self.shared else mmap.MAP_PRIVATE
        self.mem = mmap.mmap(-1, self.size, flags, mmap.PROT_READ | mmap.PROT_WRITE)


class ShmRegion(Region):
    def allocate(self):
        self._open(os.path.join(SHM_DIR, f"hostmem-{os.getpid()}-{self.index}"))


class MemfdRegion(Region):
    def allocate(self):
        self.fd = os.memfd_create(f"hostmem-{self.index}")
        os.ftruncate(self.fd, self.size)
        self.mem = self._map(self.fd)


class FileRegion(Region):
    directory = "/tmp"

    def allocate(self):
        self._open(os.path.join(self.directory, f"hostmem-{os.getpid()}-{self.index}"))


KINDS = {
    "anon": AnonRegion, "shared-anon": AnonRegion,
    "shm": ShmRegion, "ghost-shm": ShmRegion,
    "memfd": MemfdRegion,
    "file": FileRegion, "ghost-file": FileRegion,
    "sparse": FileRegion, "ghost-sparse": FileRegion,
}


def make_regions(spec):
    regions = []
    shm_free = os.statvfs(SHM_DIR).f_bavail * os.statvfs(SHM_DIR).f_frsize - SHM_RESERVE
    for index, item in enumerate(spec.split(",")):
        parts = item.strip().split(":")
        kind, size = parts[0], parse_size(parts[1])
        if kind not in KINDS:
            sys.exit(f"unknown region kind {kind!r}; one of {', '.join(KINDS)}")
        dense = parts[2] == "dense" if len(parts) > 2 else not kind.endswith("sparse")
        if KINDS[kind] is ShmRegion and size > shm_free:
            # A dense region uses all of its size; a docker /dev/shm is only 64 MiB
            if shm_free < 1 << 20:
                log(f"NOTE {kind}:{human(size)} skipped: {SHM_DIR} has {human(max(shm_free, 0))} free")
                continue
            log(f"NOTE {kind}:{human(size)} shrunk to {human(shm_free)}: that's what {SHM_DIR} has free")
            size = shm_free - shm_free % PAGE
        if size < 16 * PAGE + 1000:
            sys.exit(f"{item}: too small, at least {16 * PAGE + 1000} bytes")
        region = KINDS[kind](index, kind, size, dense)
        if KINDS[kind] is ShmRegion:
            shm_free -= region.size
        regions.append(region)
    return regions


def child(k, regions, ready_w, parent):
    """Verify every region, then count the pass in each comm page, until stopped"""
    for r in regions:
        r.remap()
        r.release(unlink=False)  # this child's fds; the parent unlinks the ghosts
    os.write(ready_w, b"x")
    os.close(ready_w)
    n = 0
    while os.getppid() == parent:
        for r in regions:
            r.verify(f"child{k}")
        n += 1
        for r in regions:
            if not r.shared and r.counter(k) != n - 1:
                fail(f"child{k}: {r.describe()}: its own private counter is {r.counter(k)}, not {n - 1}")
            r.set_counter(k, n)
        time.sleep(0.1)
    sys.exit(0)


def counts(regions, procs):
    """Each child's lowest counter across the shared regions"""
    shared = [r for r in regions if r.shared]
    if not shared:
        return [0] * procs
    return [min(r.counter(k) for r in shared) for k in range(procs)]


def check(regions, procs, pids, last):
    """One pass of the parent's: every region, the private comm pages, the children"""
    for r in regions:
        r.verify("parent")
        if not r.shared and r.mem[:PAGE] != ZERO:
            fail(f"parent: {r.describe()}: a child's write to private memory is visible")
    for k, pid in enumerate(pids):
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            fail(f"parent: child{k} exited, status {status:#x}")
    now = counts(regions, procs)
    for k in range(procs):
        if now[k] < last[k]:
            fail(f"parent: child{k}'s counter went back, {last[k]} -> {now[k]}")
    return now


def main():
    global out
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--regions", help="KIND:SIZE[:dense|sparse],... (sizes take K, M, G)")
    p.add_argument("--preset", choices=PRESETS, help="a set of regions: " +
                   "; ".join(f"{k}={v}" for k, v in PRESETS.items()))
    p.add_argument("--procs", type=int, default=3, help="children to fork (default 3)")
    p.add_argument("--dir", default="/tmp", help="for file and sparse regions (default /tmp)")
    p.add_argument("--gate", help="once this file exists: verify, print PASS and exit 0")
    p.add_argument("--log", help="append to this file instead of printing")
    args = p.parse_args()
    if args.log:
        out = open(args.log, "a", buffering=1)
    if not 1 <= args.procs <= PAGE // 8:
        sys.exit(f"--procs: 1 to {PAGE // 8}")
    FileRegion.directory = args.dir
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # clean up the kept files

    spec = ",".join(s for s in (PRESETS.get(args.preset), args.regions) if s) or PRESETS["small"]
    regions = make_regions(spec)
    if not regions:
        sys.exit("no regions to map")
    me = os.getpid()
    pids = []
    try:
        # Inside the try: a SIGTERM while allocating or filling (large presets take a while)
        # still removes the files created so far (only regions allocated have a path)
        for r in regions:
            r.allocate()
            r.fill()
            log(f"REGION {r.describe()}")

        ready_r, ready_w = os.pipe()
        for k in range(args.procs):
            pid = os.fork()
            if pid == 0:
                # Exit here, never through the parent's cleanup below
                code = 1
                try:
                    os.close(ready_r)
                    child(k, regions, ready_w, me)
                except SystemExit as e:
                    code = e.code if isinstance(e.code, int) else 1
                except Exception as e:
                    log(f"MISMATCH child{k}: {e!r}")
                finally:
                    os._exit(code)
            pids.append(pid)
        os.close(ready_w)
        for _ in range(args.procs):
            if not os.read(ready_r, 1):
                fail("parent: a child exited before it was ready")
        os.close(ready_r)
        for r in regions:
            r.release()  # now the ghosts and the memfd live on in the mappings alone
        log(f"READY {len(regions)} regions, {args.procs} children")

        last = [0] * args.procs
        while not (args.gate and os.path.exists(args.gate)):
            last = check(regions, args.procs, pids, last)
            log("OK " + " ".join(map(str, last)))
            time.sleep(0.5)

        # The gate: everything once more, and every child still writing to every shared region
        last = check(regions, args.procs, pids, last)
        deadline = time.monotonic() + 30
        while any(r.shared for r in regions) and \
                any(c <= l for c, l in zip(counts(regions, args.procs), last)):
            if time.monotonic() > deadline:
                fail(f"parent: the children's counters stopped at {counts(regions, args.procs)}")
            time.sleep(0.1)
        check(regions, args.procs, pids, last)
        for pid in pids:
            os.kill(pid, signal.SIGTERM)
            os.waitpid(pid, 0)
        pids = []
        log("PASS")
    finally:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for r in regions:  # the kept files
            if r.path and not r.path.endswith("(unlinked)") and os.path.exists(r.path):
                os.unlink(r.path)


if __name__ == "__main__":
    main()
