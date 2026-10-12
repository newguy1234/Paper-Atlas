"""Shared output destination for the UI, terminal and MCP tools."""
import os
from pathlib import Path


def downloads_folder() -> Path:
    if os.name == 'nt':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as key:
                value, _ = winreg.QueryValueEx(key, '{374DE290-123F-4565-9164-39C4925E467B}')
                return Path(os.path.expandvars(value)).expanduser()
        except OSError:
            pass
    return Path.home() / 'Downloads'


def output_folder(value=None, store=None) -> Path:
    # An explicit blank resets to Downloads; None uses the saved preference.
    if value is None and store is not None:
        value = store.get_setting('pdf_output_dir', '')
    value = str(value or '').strip()
    return (Path(os.path.expandvars(value)).expanduser() if value else downloads_folder()).resolve()
