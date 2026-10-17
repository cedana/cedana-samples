#!/usr/bin/env python3
"""
File descriptors of the kinds servers hold, each with state in flight, in one process tree.

Each kind, and what is in flight between checks:

  epoll       an epoll set over fds of the other kinds (level- and edge-triggered, EPOLLIN
              and EPOLLOUT, and an idle pipe): the registrations in /proc/self/fdinfo must
              be unchanged, and exactly the fds that have data pending must be ready
  eventfd     a pending count
  timerfd     an armed periodic timer (10 ms): still armed with the same interval, and ticking
  signalfd    blocked signals pending for it to read: SIGUSR1 and two queued SIGRTMIN+1
  inotify     a watch on a directory, with IN_CREATE and IN_MOVED_TO events queued
  pipe        unread bytes in a pipe
  fifo        unread bytes in a named pipe in --dir
  stream      a SOCK_STREAM socketpair with unread bytes in both directions
  seqpacket   a SOCK_SEQPACKET socketpair with unread messages in both directions (boundaries
              must hold)
  udp         two datagrams queued on a UDP socket on 127.0.0.1, from a known address
  pty         a pty pair in raw mode, with unread bytes in both directions
  unix        an abstract-namespace unix listener, connected to a client in a child process
  tcp         a TCP listener on 127.0.0.1, with an established connection to the same child

The child is the client end of unix and tcp: it and the parent each keep one message unread
in the other's direction, so both directions carry unread bytes. Every 0.5 s the parent checks
everything in flight (consuming it), sends the next round, and logs `OK <round>`. Lost,
extra or corrupted state is logged as `MISMATCH ...` and the process exits 1, as does the
child dying. The parent logs `FD ...` lines with the fd numbers, then READY.

With --gate PATH, once the file exists the parent checks one more round, requires the timer
to tick and both listeners to accept a new connection, stops the child, prints PASS and exits
0, so a test can checkpoint and restore it mid-run, then create the gate. Without it, it runs
until killed. tcp needs established TCP connections to be checkpointed (CRIU's
--tcp-established); --skip leaves out kinds. Stdlib only, no root needed. File locks are not
covered here.

Usage: fd_zoo.py [--gate PATH] [--log PATH] [--dir DIR] [--skip KIND,...]
"""

import argparse
import ctypes
import fcntl
import os
import re
import select
import shutil
import signal
import socket
import struct
import sys
import tempfile
import termios
import time
import tty

TICK = 0.5
POLL = 0.05  # one wait while reading bytes that should be there
LOCAL_POLLS = 40  # data from this process must be readable within this many waits (2 s)
PEER_POLLS = 400  # data from the child within this many (20 s); counted, so a checkpoint
#                   in the middle of a wait costs one wait, not the whole timeout
TIMER_NS = 10_000_000
SIG_STD, SIG_RT = signal.SIGUSR1, signal.SIGRTMIN + 1

KINDS = ["epoll", "eventfd", "timerfd", "signalfd", "inotify", "pipe", "fifo", "stream",
         "seqpacket", "udp", "pty", "unix", "tcp"]

out = sys.stdout
libc = ctypes.CDLL(None, use_errno=True)


def log(msg):
    out.write(msg + "\n")
    out.flush()


def fail(msg):
    log(f"MISMATCH {msg}")
    sys.exit(1)


def payload(tag, seq, size):
    unit = f"{tag}:{seq};".encode()
    return (unit * (size // len(unit) + 1))[:size]


def pending(fd):
    """Bytes readable without blocking (FIONREAD works for pipes, ttys and sockets)"""
    return struct.unpack("i", fcntl.ioctl(fd, termios.FIONREAD, b"\0" * 4))[0]


def read_exact(fd, n, what, polls=LOCAL_POLLS):
    """n bytes from a stream, waiting a bounded number of times for them"""
    data = b""
    waits = 0
    while len(data) < n:
        r, _, _ = select.select([fd], [], [], POLL)
        if not r:
            waits += 1
            if waits > polls:
                fail(f"{what}: {len(data)} of {n} bytes arrived")
            continue
        chunk = os.read(fd, n - len(data))
        if not chunk:
            fail(f"{what}: closed after {len(data)} of {n} bytes")
        data += chunk
    return data


def expect(what, got, want):
    if got != want:
        show = (lambda b: b[:40] + (b"..." if len(b) > 40 else b"")) if isinstance(got, bytes) else repr
        fail(f"{what}: got {show(got)}, expected {show(want)}")


def expect_drained(fd, what):
    n = pending(fd)
    if n:
        fail(f"{what}: {n} unexpected bytes after the expected ones")


def fdinfo(fd):
    with open(f"/proc/self/fdinfo/{fd}") as f:
        return f.read()


def check_errno(rc, what):
    if rc < 0:
        e = ctypes.get_errno()
        raise OSError(e, f"{what}: {os.strerror(e)}")
    return rc


class Kind:
    """One kind of fd: refill(seq) puts round seq's state in flight, check(seq) consumes and
    verifies it. epoll() lists (fd, events, ready?) for the epoll set, ready None: don't care."""

    name = ""

    def setup(self):
        pass

    def describe(self):
        return ""

    def refill(self, seq):
        pass

    def check(self, seq):
        pass

    def epoll(self):
        return []

    def final(self):
        """At the gate, after one more check: anything that needs time or a new action"""

    def cleanup(self):
        pass


class Eventfd(Kind):
    name = "eventfd"

    def setup(self):
        if not hasattr(os, "eventfd"):
            return "os.eventfd needs Python 3.10"
        self.fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

    def describe(self):
        return f"fd={self.fd}"

    def refill(self, seq):
        os.eventfd_write(self.fd, 1000 + seq)
        os.eventfd_write(self.fd, 7)

    def check(self, seq):
        try:
            got = os.eventfd_read(self.fd)
        except BlockingIOError:
            fail(f"eventfd: count 0, expected {1007 + seq}")
        expect("eventfd: count", got, 1007 + seq)

    def epoll(self):
        return [(self.fd, select.EPOLLIN, True)]


class Timerfd(Kind):
    name = "timerfd"

    class itimerspec(ctypes.Structure):
        _fields_ = [("it_interval", ctypes.c_long * 2), ("it_value", ctypes.c_long * 2)]

    def setup(self):
        flags = os.O_NONBLOCK | os.O_CLOEXEC  # TFD_NONBLOCK, TFD_CLOEXEC
        if hasattr(os, "timerfd_create"):  # Python 3.13
            self.fd = os.timerfd_create(time.CLOCK_MONOTONIC, flags=flags)
            os.timerfd_settime_ns(self.fd, initial=TIMER_NS, interval=TIMER_NS)
        else:
            if not hasattr(libc, "timerfd_create"):
                return "no timerfd_create in this libc"
            self.fd = check_errno(libc.timerfd_create(time.CLOCK_MONOTONIC, flags), "timerfd_create")
            spec = self.itimerspec((0, TIMER_NS), (0, TIMER_NS))
            check_errno(libc.timerfd_settime(self.fd, 0, ctypes.byref(spec), None), "timerfd_settime")
        self.ticks = 0

    def describe(self):
        return f"fd={self.fd} interval={TIMER_NS // 1000000}ms"

    def read_ticks(self):
        try:
            n = struct.unpack("Q", os.read(self.fd, 8))[0]
        except BlockingIOError:
            n = 0
        self.ticks += n
        return n

    def check(self, seq):
        # From fdinfo, the same on every Python. A disarmed timer has an interval of (0, 0);
        # it_value reads 0 while a tick is unread, so it says nothing.
        interval = re.search(r"it_interval:\s*\((\d+),\s*(\d+)\)", fdinfo(self.fd))
        if not interval:
            fail("timerfd: no it_interval in fdinfo")
        expect("timerfd: interval (s, ns)", (int(interval[1]), int(interval[2])), (0, TIMER_NS))
        self.read_ticks()

    def epoll(self):
        return [(self.fd, select.EPOLLIN, None)]

    def final(self):
        before = self.ticks
        for _ in range(LOCAL_POLLS):
            if self.read_ticks():
                return
            time.sleep(POLL)
        fail(f"timerfd: no tick after the gate ({before} before)")


class Signalfd(Kind):
    name = "signalfd"
    SIGINFO = 128  # sizeof(struct signalfd_siginfo); ssi_signo is the first u32

    def setup(self):
        if not hasattr(libc, "signalfd"):
            return "no signalfd in this libc"
        words = 1024 // (8 * ctypes.sizeof(ctypes.c_ulong))
        mask = (ctypes.c_ulong * words)()
        for s in (SIG_STD, SIG_RT):
            mask[(s - 1) // 64] |= 1 << ((s - 1) % 64)
        signal.pthread_sigmask(signal.SIG_BLOCK, {SIG_STD, SIG_RT})
        self.fd = check_errno(libc.signalfd(-1, ctypes.byref(mask), os.O_NONBLOCK | os.O_CLOEXEC),
                              "signalfd")

    def describe(self):
        return f"fd={self.fd} signals={int(SIG_STD)},{int(SIG_RT)}"

    def refill(self, seq):
        # A standard signal pends once; a real-time one queues each time
        os.kill(os.getpid(), SIG_STD)
        os.kill(os.getpid(), SIG_RT)
        os.kill(os.getpid(), SIG_RT)

    def check(self, seq):
        got = []
        while True:
            try:
                info = os.read(self.fd, self.SIGINFO)
            except BlockingIOError:
                break
            signo, _, _, pid = struct.unpack_from("IiiI", info)
            if pid != os.getpid():
                fail(f"signalfd: signal {signo} from pid {pid}, expected {os.getpid()}")
            got.append(signo)
        expect("signalfd: pending signals, in order", got, [int(SIG_STD), int(SIG_RT), int(SIG_RT)])

    def epoll(self):
        return [(self.fd, select.EPOLLIN, True)]


class Inotify(Kind):
    name = "inotify"
    IN_MOVED_TO, IN_CREATE = 0x80, 0x100

    def setup(self):
        if not hasattr(libc, "inotify_init1"):
            return "no inotify_init1 in this libc"
        self.dir = os.path.join(Zoo.dir, "watched")
        os.mkdir(self.dir)
        self.fd = check_errno(libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC), "inotify_init1")
        self.wd = check_errno(libc.inotify_add_watch(self.fd, self.dir.encode(),
                                                     self.IN_CREATE | self.IN_MOVED_TO),
                              "inotify_add_watch")

    def describe(self):
        return f"fd={self.fd} dir={self.dir}"

    def refill(self, seq):
        # IN_CREATE for a-<seq>, then IN_MOVED_TO for b-<seq>; unlinking it later is not watched
        a = os.path.join(self.dir, f"a-{seq}")
        os.close(os.open(a, os.O_CREAT | os.O_WRONLY, 0o600))
        os.rename(a, os.path.join(self.dir, f"b-{seq}"))

    def check(self, seq):
        try:
            buf = os.read(self.fd, 65536)
        except BlockingIOError:
            buf = b""
        got, off = [], 0
        while off < len(buf):
            wd, mask, _, size = struct.unpack_from("iIII", buf, off)
            name = buf[off + 16:off + 16 + size].rstrip(b"\0").decode()
            got.append((wd, mask, name))
            off += 16 + size
        expect("inotify: events", got, [(self.wd, self.IN_CREATE, f"a-{seq}"),
                                        (self.wd, self.IN_MOVED_TO, f"b-{seq}")])
        os.unlink(os.path.join(self.dir, f"b-{seq}"))

    def epoll(self):
        return [(self.fd, select.EPOLLIN, True)]


class Pipe(Kind):
    name = "pipe"
    SIZE = 3000

    def setup(self):
        self.r, self.w = os.pipe()

    def describe(self):
        return f"r={self.r} w={self.w}"

    def refill(self, seq):
        os.write(self.w, payload(self.name, seq, self.SIZE))

    def check(self, seq):
        expect(self.name, read_exact(self.r, self.SIZE, self.name), payload(self.name, seq, self.SIZE))
        expect_drained(self.r, self.name)

    def epoll(self):
        return [(self.r, select.EPOLLIN, True), (self.w, select.EPOLLOUT, True)]


class Fifo(Pipe):
    name = "fifo"

    def setup(self):
        self.path = os.path.join(Zoo.dir, "fifo")
        os.mkfifo(self.path, 0o600)
        self.r = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        self.w = os.open(self.path, os.O_WRONLY)

    def describe(self):
        return f"r={self.r} w={self.w} path={self.path}"

    def epoll(self):
        return [(self.r, select.EPOLLIN, True)]


class Stream(Kind):
    name = "stream"
    TYPE = socket.SOCK_STREAM

    def setup(self):
        self.a, self.b = socket.socketpair(socket.AF_UNIX, self.TYPE)

    def describe(self):
        return f"a={self.a.fileno()} b={self.b.fileno()}"

    def refill(self, seq):
        self.a.sendall(payload("a>b", seq, 5000))
        self.b.sendall(payload("b>a", seq, 2000))

    def check(self, seq):
        for sock, tag, size in ((self.b, "a>b", 5000), (self.a, "b>a", 2000)):
            what = f"stream {tag}"
            expect(what, read_exact(sock.fileno(), size, what), payload(tag, seq, size))
            expect_drained(sock.fileno(), what)

    def epoll(self):
        # b edge-triggered: whether it shows as ready depends on earlier polls, so don't care
        return [(self.a.fileno(), select.EPOLLIN, True),
                (self.b.fileno(), select.EPOLLIN | select.EPOLLET, None)]


class Seqpacket(Stream):
    name = "seqpacket"
    TYPE = socket.SOCK_SEQPACKET
    MESSAGES = (("a>b", 1500), ("a>b", 300), ("b>a", 700))

    def refill(self, seq):
        for tag, size in self.MESSAGES:
            (self.a if tag == "a>b" else self.b).send(payload(tag, seq, size))

    def check(self, seq):
        for tag, size in self.MESSAGES:
            sock = self.b if tag == "a>b" else self.a
            try:
                got = sock.recv(65536, socket.MSG_DONTWAIT)
            except BlockingIOError:
                fail(f"seqpacket {tag}: a {size}-byte message is missing")
            expect(f"seqpacket {tag} message", got, payload(tag, seq, size))
        for sock in (self.a, self.b):
            try:
                extra = sock.recv(65536, socket.MSG_DONTWAIT)
                fail(f"seqpacket: an unexpected {len(extra)}-byte message")
            except BlockingIOError:
                pass

    def epoll(self):
        return [(self.b.fileno(), select.EPOLLIN, True)]


class Udp(Kind):
    name = "udp"
    SIZES = (1200, 64)

    def setup(self):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("127.0.0.1", 0))
        self.tx.bind(("127.0.0.1", 0))
        self.rx.setblocking(False)

    def describe(self):
        return f"rx={self.rx.fileno()} at {self.rx.getsockname()[1]} tx={self.tx.fileno()}"

    def refill(self, seq):
        for i, size in enumerate(self.SIZES):
            self.tx.sendto(payload(f"udp{i}", seq, size), self.rx.getsockname())

    def check(self, seq):
        for i, size in enumerate(self.SIZES):
            for _ in range(LOCAL_POLLS):
                try:
                    got, src = self.rx.recvfrom(65536)
                    break
                except BlockingIOError:
                    time.sleep(POLL)
            else:
                fail(f"udp: datagram {i} of round {seq} is missing")
            expect(f"udp: datagram {i}", got, payload(f"udp{i}", seq, size))
            expect("udp: source", src, self.tx.getsockname())
        try:
            extra = self.rx.recv(65536)
            fail(f"udp: an unexpected {len(extra)}-byte datagram")
        except BlockingIOError:
            pass

    def epoll(self):
        return [(self.rx.fileno(), select.EPOLLIN, True)]


class Pty(Kind):
    name = "pty"

    def setup(self):
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)  # bytes pass as they are: no echo, no line editing, no \r\n

    def describe(self):
        return f"master={self.master} slave={self.slave} ({os.ttyname(self.slave)})"

    def refill(self, seq):
        os.write(self.master, payload("m>s", seq, 600))
        os.write(self.slave, payload("s>m", seq, 300))

    def check(self, seq):
        for fd, tag, size in ((self.slave, "m>s", 600), (self.master, "s>m", 300)):
            what = f"pty {tag}"
            expect(what, read_exact(fd, size, what), payload(tag, seq, size))
            expect_drained(fd, what)

    def epoll(self):
        return [(self.slave, select.EPOLLIN, None)]  # a pty passes data on asynchronously


class Connection(Kind):
    """A listener and a connection to it, whose client end is in the child. Each side keeps
    one message unread in the other's direction: the parent sends s>c n+1, then reads c>s n."""

    SIZE = 3000

    def listener(self):
        raise NotImplementedError

    def setup(self):
        self.lsock = self.listener()
        self.lsock.listen(8)
        self.client = socket.socket(self.lsock.family, socket.SOCK_STREAM)
        self.client.connect(self.lsock.getsockname())
        self.conn, _ = self.lsock.accept()

    def describe(self):
        return f"listener={self.lsock.fileno()} at {self.address()} server={self.conn.fileno()}"

    def address(self):
        return self.lsock.getsockname()

    def after_fork(self):
        """In the parent: the client end belongs to the child alone"""
        self.client.close()

    def refill(self, seq):
        self.conn.sendall(payload(f"{self.name}s>c", seq, self.SIZE))

    def check(self, seq):
        what = f"{self.name} c>s from the child"
        expect(what, read_exact(self.conn.fileno(), self.SIZE, what, PEER_POLLS),
               payload(f"{self.name}c>s", seq, self.SIZE))

    def client_send(self, seq):
        self.client.sendall(payload(f"{self.name}c>s", seq, self.SIZE))

    def client_check(self, seq):
        what = f"child: {self.name} s>c"
        expect(what, read_exact(self.client.fileno(), self.SIZE, what, PEER_POLLS),
               payload(f"{self.name}s>c", seq, self.SIZE))

    def epoll(self):
        return [(self.conn.fileno(), select.EPOLLIN, None)]

    def final(self):
        # The listener still accepts, at the same address
        c = socket.socket(self.lsock.family, socket.SOCK_STREAM)
        c.settimeout(5)
        c.connect(self.lsock.getsockname())
        self.lsock.settimeout(5)
        s, _ = self.lsock.accept()
        c.sendall(b"hello")
        s.settimeout(5)
        expect(f"{self.name}: a new connection", s.recv(5), b"hello")
        c.close()
        s.close()


class Unix(Connection):
    name = "unix"

    def listener(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(f"\0fd_zoo-{os.getpid()}")
        return s

    def address(self):
        return "@" + self.lsock.getsockname()[1:].decode()


class Tcp(Connection):
    name = "tcp"
    SIZE = 32768

    def listener(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        return s

    def address(self):
        host, port = self.lsock.getsockname()
        return f"{host}:{port}"


class Epoll(Kind):
    name = "epoll"

    def __init__(self, others):
        self.others = others

    def setup(self):
        self.ep = select.epoll()
        self.idle_r, self.idle_w = os.pipe()  # never written: must never be ready
        self.want = [(self.idle_r, select.EPOLLIN, False)]
        self.names = {self.idle_r: "idle pipe"}
        for k in self.others:
            self.want += k.epoll()
            self.names.update((fd, k.name) for fd, _, _ in k.epoll())
        for fd, events, _ in self.want:
            self.ep.register(fd, events)

    def describe(self):
        return f"fd={self.ep.fileno()} watching {len(self.want)} fds"

    def check(self, seq):
        # The registrations: fd, and events less the EPOLLERR|EPOLLHUP the kernel adds
        found = {int(fd): int(ev, 16) & ~(select.EPOLLERR | select.EPOLLHUP)
                 for fd, ev in re.findall(r"tfd:\s*(\d+)\s+events:\s*([0-9a-f]+)", fdinfo(self.ep.fileno()))}
        expect("epoll: registrations {fd: events}", found, {fd: ev for fd, ev, _ in self.want})
        ready = dict(self.ep.poll(0))
        for fd, events, want in self.want:
            if want is None:
                continue
            got = ready.get(fd, 0) & events
            if bool(got) != want:
                fail(f"epoll: fd {fd} ({self.names[fd]}) {'not ' if want else ''}ready, "
                     f"events {ready.get(fd, 0):#x}")


class Zoo:
    dir = "/tmp"


def child(conns, parent):
    """The client ends of the connections: read the server's message n, send ours n+1"""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    seq = 0
    while os.getppid() == parent:
        for c in conns:
            c.client_check(seq)
            c.client_send(seq + 1)
        seq += 1
        time.sleep(TICK)


def main():
    global out
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gate", help="once this file exists: verify, print PASS and exit 0")
    p.add_argument("--log", help="append to this file instead of printing")
    p.add_argument("--dir", default="/tmp", help="for the fifo and the watched directory (default /tmp)")
    p.add_argument("--skip", default="", help="kinds to leave out: " + ",".join(KINDS))
    args = p.parse_args()
    if args.log:
        out = open(args.log, "a", buffering=1)
    skip = set(filter(None, args.skip.split(",")))
    if skip - set(KINDS):
        sys.exit(f"--skip: unknown kinds {','.join(sorted(skip - set(KINDS)))}; one of {','.join(KINDS)}")
    Zoo.dir = tempfile.mkdtemp(prefix="fd_zoo-", dir=args.dir)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # clean up --dir

    pid = None
    try:
        classes = [Eventfd, Timerfd, Signalfd, Inotify, Pipe, Fifo, Stream, Seqpacket, Udp, Pty,
                   Unix, Tcp]
        kinds = []
        for cls in classes:
            if cls.name in skip:
                continue
            k = cls()
            why = k.setup()
            if why:
                log(f"NOTE {k.name} skipped: {why}")
                continue
            kinds.append(k)
        if "epoll" not in skip:
            ep = Epoll(kinds)
            ep.setup()
            kinds.insert(0, ep)  # first: it looks at the others' state before they consume it
        conns = [k for k in kinds if isinstance(k, Connection)]

        me = os.getpid()
        for c in conns:  # message 0 each way, before the child starts
            c.refill(0)
            c.client_send(0)
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                child(conns, me)
                code = 0
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
            except Exception as e:
                log(f"MISMATCH child: {e!r}")
            finally:
                os._exit(code if os.getppid() == me else 0)  # the parent reports its own end
        for c in conns:
            c.after_fork()

        for k in kinds:
            if not isinstance(k, Connection):
                k.refill(0)
            log(f"FD {k.name} {k.describe()}")
        log(f"READY {len(kinds)} kinds, pid {me}, child {pid}")

        seq = 0
        while True:
            done = bool(args.gate) and os.path.exists(args.gate)  # this round is the last
            if os.waitpid(pid, os.WNOHANG)[0]:
                pid = None
                fail("the child exited")
            for k in kinds:
                k.check(seq)
                k.refill(seq + 1)
            seq += 1
            log(f"OK {seq}")
            if done:
                break
            time.sleep(TICK)

        for k in kinds:
            k.final()
        if os.waitpid(pid, os.WNOHANG)[0]:
            pid = None
            fail("the child exited")
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)
        pid = None
        log(f"PASS {seq} rounds")
    except OSError as e:
        fail(f"{e!r}")
    finally:
        if pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        shutil.rmtree(Zoo.dir, ignore_errors=True)


if __name__ == "__main__":
    main()
