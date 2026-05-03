"""Encrypt API keys at rest for queued jobs (Fernet)."""

from cryptography.fernet import Fernet, InvalidToken

from .settings import get_settings


def _fernet() -> Fernet:
    return Fernet(get_settings().encryption_key.encode())


def encrypt_secret(plain: str) -> bytes:
    return _fernet().encrypt(plain.encode())


def decrypt_secret(blob: bytes) -> str:
    try:
        return _fernet().decrypt(blob).decode()
    except InvalidToken as e:
        raise ValueError("invalid or corrupted ciphertext") from e
