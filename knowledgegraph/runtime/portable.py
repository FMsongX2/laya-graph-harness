"""POSIX/Windows boundaries for locks, interpreters, environments and owned processes."""
from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys

WINDOWS = os.name == 'nt'

# Variables a child needs to start at all, beyond the platform-neutral ones the callers choose.
# Windows Python cannot create sockets or find system DLLs without SYSTEMROOT, and CUDA builds
# locate the driver/toolkit through CUDA_PATH and honour CUDA_VISIBLE_DEVICES.
PLATFORM_ENVIRONMENT = (('SYSTEMROOT', 'SYSTEMDRIVE', 'WINDIR', 'COMSPEC', 'PATHEXT', 'USERPROFILE',
                         'USERNAME', 'APPDATA', 'LOCALAPPDATA', 'PROGRAMDATA', 'PROGRAMFILES',
                         'PROGRAMFILES(X86)', 'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE',
                         'TEMP', 'TMP') if WINDOWS else ()) + \
                       ('CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'PYTORCH_CUDA_ALLOC_CONF')


def passthrough_environment(keys):
    """Copy only the named variables plus platform essentials; JSON/text files are UTF-8 everywhere."""
    env = {key: os.environ[key] for key in (*keys, *PLATFORM_ENVIRONMENT) if key in os.environ}
    env['PYTHONUTF8'] = '1'
    return env


def temporary_environment(path):
    """Point every platform's temporary-directory variable at one owned directory."""
    return {'TMPDIR': str(path), **({'TEMP': str(path), 'TMP': str(path)} if WINDOWS else {})}


def venv_python(venv):
    return Path(venv) / ('Scripts/python.exe' if WINDOWS else 'bin/python')


def interpreter(path):
    """Accept a configured POSIX venv path (…/bin/python) on Windows as well."""
    path = Path(path)
    if WINDOWS and path.parent.name == 'bin' and path.name.startswith('python'):
        return path.parent.parent / 'Scripts' / 'python.exe'
    return path


# ---------------------------------------------------------------- advisory file locks
if WINDOWS:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class _Overlapped(ctypes.Structure):
        _fields_ = [('Internal', ctypes.c_void_p), ('InternalHigh', ctypes.c_void_p),
                    ('Offset', wintypes.DWORD), ('OffsetHigh', wintypes.DWORD), ('hEvent', wintypes.HANDLE)]

    _kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel32.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_Overlapped)]
    _kernel32.LockFileEx.restype = wintypes.BOOL
    _kernel32.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                       wintypes.DWORD, ctypes.POINTER(_Overlapped)]
    _kernel32.UnlockFileEx.restype = wintypes.BOOL
    _EXCLUSIVE, _FAIL_IMMEDIATELY, _LOCK_VIOLATION, _NOT_LOCKED = 0x2, 0x1, 33, 158
    # Lock one byte far past any content: Windows byte-range locks are mandatory, so this keeps
    # them advisory for writers of the (normally empty) lock file, like flock on POSIX.
    _REGION = (0, 1, 0, 0x7FFFFFFF)
else:
    import fcntl


def lock(handle, *, shared=False, blocking=True):
    """flock-equivalent per open file; raises BlockingIOError when non-blocking and held."""
    if not WINDOWS:
        fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if blocking else fcntl.LOCK_NB))
        return
    flags = (0 if shared else _EXCLUSIVE) | (0 if blocking else _FAIL_IMMEDIATELY)
    reserved, low, high, offset_high = _REGION
    overlapped = _Overlapped(OffsetHigh=offset_high)
    if not _kernel32.LockFileEx(msvcrt.get_osfhandle(handle.fileno()), flags, reserved, low, high,
                                ctypes.byref(overlapped)):
        error = ctypes.get_last_error()
        if error == _LOCK_VIOLATION: raise BlockingIOError('File lock is held by another owner')
        raise ctypes.WinError(error)


def unlock(handle):
    if not WINDOWS:
        fcntl.flock(handle, fcntl.LOCK_UN)
        return
    reserved, low, high, offset_high = _REGION
    overlapped = _Overlapped(OffsetHigh=offset_high)
    if not _kernel32.UnlockFileEx(msvcrt.get_osfhandle(handle.fileno()), reserved, low, high,
                                  ctypes.byref(overlapped)):
        error = ctypes.get_last_error()
        if error != _NOT_LOCKED: raise ctypes.WinError(error)


@contextmanager
def locked(path, *, shared=False, blocking=True, mode='a'):
    """Hold a lock on `path` and release it explicitly before closing the handle."""
    with Path(path).open(mode) as handle:
        lock(handle, shared=shared, blocking=blocking)
        try: yield handle
        finally: unlock(handle)


# ---------------------------------------------------------------- owned processes
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000
_DETACHED_PROCESS = 0x00000008


def detached():
    """Popen arguments for a service that outlives its launcher in its own signal group.

    Windows: a new process group with its own hidden console, so the launching terminal can
    close and CTRL_BREAK can later be delivered to exactly this group.
    """
    if WINDOWS: return {'creationflags': _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW}
    return {'start_new_session': True}


def isolated_child():
    """Popen arguments for a helper that must not receive its parent's stop signal."""
    return {'creationflags': _CREATE_NEW_PROCESS_GROUP} if WINDOWS else {}


def command_line(pid):
    """The process's command line, or '' when it no longer exists."""
    if not WINDOWS:
        return subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout.strip()
    import psutil
    try: return subprocess.list2cmdline(psutil.Process(pid).cmdline())
    except (psutil.Error, OSError): return ''


def is_group_leader(pid):
    """POSIX: the PID leads its own process group. Windows: groups are fixed at creation by
    `detached()`, so a live owned PID is the leader of the tree it started."""
    if WINDOWS: return bool(command_line(pid))
    try: return os.getpgid(pid) == pid
    except ProcessLookupError: return False


# CTRL_BREAK reaches only processes attached to the same console. A detached service owns a
# hidden console, so a short helper attaches to it and signals that service's group alone.
_BREAK_HELPER = r'''
import ctypes, sys
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
pid = int(sys.argv[1])
kernel32.FreeConsole()
if not kernel32.AttachConsole(pid): sys.exit(3)
kernel32.SetConsoleCtrlHandler(None, True)
sent = kernel32.GenerateConsoleCtrlEvent(1, pid)
kernel32.FreeConsole()
sys.exit(0 if sent else 4)
'''


def request_stop(pid):
    """Graceful stop: SIGTERM on POSIX, CTRL_BREAK (SIGBREAK) to the owned group on Windows."""
    if not WINDOWS:
        os.kill(pid, signal.SIGTERM)
        return
    # The base interpreter avoids a venv launcher (and a console of its own) for the helper.
    python = getattr(sys, '_base_executable', None) or sys.executable
    result = subprocess.run([python, '-I', '-c', _BREAK_HELPER, str(pid)], stdin=subprocess.DEVNULL,
                            capture_output=True, timeout=15,
                            creationflags=_DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP)
    if result.returncode: raise RuntimeError(f'Could not deliver a stop request to process {pid}')


def kill_group(pid):
    """Force-stop an owned process group (POSIX) or process tree (Windows)."""
    if not WINDOWS:
        os.killpg(pid, signal.SIGKILL)
        return
    import psutil
    try: root = psutil.Process(pid)
    except psutil.NoSuchProcess: return
    tree = [*root.children(recursive=True), root]
    for process in tree:
        try: process.kill()
        except psutil.NoSuchProcess: pass
    psutil.wait_procs(tree, timeout=5)


def link_directory(link, target):
    """Directory symlink, or an NTFS junction where Windows symlinks need extra privileges."""
    try:
        Path(link).symlink_to(target, target_is_directory=True)
    except OSError:
        if not WINDOWS: raise
        import _winapi
        _winapi.CreateJunction(str(Path(target).resolve()), str(link))
