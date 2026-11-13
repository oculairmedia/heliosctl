"""Local bounded control. No SDK or microphone access in this module."""
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import stat
import time


def paths():
    root = Path(os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}'))
    info = root.stat()
    if info.st_uid != os.getuid() or not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise ValueError('runtime directory must be owned by this user and mode 0700')
    return root / 'heliosctl.sock', root / 'heliosctl.receipt'


class Control:
    def __init__(self, gain, depth, frequency, base_size=0.0, expansion=1.0):
        self.path, self.receipt = paths()
        self.params = {'gain': gain, 'ripple-depth': depth, 'frequency': frequency,
                       'base-size': base_size, 'expansion': expansion}
        self.deadline = None
        self.stopping = False
        self.clients = []
        self.server = None
        lockpath = str(self.path) + '.lock'
        fd = os.open(lockpath, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self.lock = os.fdopen(fd, 'w')
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            self.lock.close()
            raise ValueError('unsafe controller lock')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Never delete an existing endpoint: it may belong to another process.
            if os.path.lexists(self.path):
                raise ValueError('control socket exists; refuse replacement (inspect stale endpoint manually)')
            self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server.bind(str(self.path))
            os.chmod(self.path, 0o600)
            self.inode = self.path.lstat().st_ino
            self.server.listen(8)
            self.server.setblocking(False)
        except BaseException:
            if self.server:
                self.server.close()
            self.lock.close()
            raise

    def command(self, msg, now):
        if not isinstance(msg, dict):
            raise ValueError('object required')
        cmd = msg.get('command')
        if cmd == 'tune' and set(msg) == {'command', 'params'}:
            changes = msg['params']
            if not isinstance(changes, dict) or not changes or set(changes) - self.params.keys():
                raise ValueError('unknown or empty parameters')
            for key, value in changes.items():
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('finite numeric parameters required')
                if key == 'frequency' and (type(value) is not int or not 1 <= value <= 8):
                    raise ValueError('frequency must be integer 1..8')
                if key == 'gain' and not 0 < value <= 32:
                    raise ValueError('gain must be >0 and <=32')
                if key == 'ripple-depth' and not 0 <= value <= .10:
                    raise ValueError('ripple-depth must be 0..0.10')
                if key in ('base-size', 'expansion') and not 0 <= value <= 1:
                    raise ValueError('base-size and expansion must be 0..1')
            updated = dict(self.params, **changes)
            if updated['base-size'] + updated['expansion'] > 1:
                raise ValueError('base-size + expansion must not exceed 1')
            self.params = updated
        elif cmd == 'renew' and set(msg) == {'command', 'seconds'}:
            seconds = msg['seconds']
            if type(seconds) not in (float, int) or not math.isfinite(seconds) or not 0 < seconds <= 300:
                raise ValueError('renew seconds must be >0 and <=300')
            if self.stopping or self.deadline is None or now >= self.deadline:
                raise ValueError('session not renewable')
            self.deadline = now + seconds
        elif cmd == 'stop' and set(msg) == {'command'}:
            self.stopping = True
        elif cmd != 'status' or set(msg) != {'command'}:
            raise ValueError('invalid command')
        return {'state': 'stopping' if self.stopping else 'running', 'pid': os.getpid(),
                'params': self.params, 'remaining': None if self.deadline == math.inf else max(0, (self.deadline or now) - now)}

    def poll(self, now):
        # Fixed work budget; slow/EOF/flood clients cannot block the frame path.
        for _ in range(4):
            try:
                conn, _ = self.server.accept()
            except BlockingIOError:
                break
            conn.setblocking(False)
            if len(self.clients) >= 8:
                conn.close()
            else:
                self.clients.append((conn, b'', now + .5))
        pending = []
        for conn, data, expiry in self.clients:
            try:
                if now >= expiry:
                    conn.close()
                    continue
                try:
                    chunk = conn.recv(2049 - len(data))
                except BlockingIOError:
                    pending.append((conn, data, expiry))
                    continue
                data += chunk
                if len(data) > 2048 or not chunk:
                    conn.close()
                    continue
                if b'\n' not in data:
                    pending.append((conn, data, expiry))
                    continue
                try:
                    result = self.command(json.loads(data), now)
                except (ValueError, TypeError) as exc:
                    result = {'error': str(exc)}
                conn.send(json.dumps(result).encode() + b'\n')
            except OSError:
                pass
            finally:
                if not any(item[0] is conn for item in pending):
                    conn.close()
        self.clients = pending

    def finish(self, error):
        # Called only after SDK cleanup and microphone reap have completed.
        result = {'state': 'failed' if error else 'stopped', 'pid': os.getpid(), 'error': error}
        tmp = str(self.receipt) + f'.{os.getpid()}.tmp'
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as out:
            json.dump(result, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, self.receipt)
        for conn, _, _ in self.clients:
            conn.close()
        self.server.close()
        if self.path.lstat().st_ino == self.inode:
            self.path.unlink()
        self.lock.close()


def request(message):
    path, receipt = paths()
    if not path.exists() and message == {'command': 'status'}:
        return json.loads(receipt.read_text())
    info = path.lstat()
    if info.st_uid != os.getuid() or not stat.S_ISSOCK(info.st_mode) or info.st_mode & 0o077:
        raise ValueError('unsafe controller endpoint')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(2)
        conn.connect(str(path))
        conn.sendall(json.dumps(message).encode() + b'\n')
        data = conn.recv(4096)
        return json.loads(data)
