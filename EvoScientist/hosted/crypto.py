"""Encrypt API keys at rest (Postgres) before workers pick up jobs."""

from cryptography.fernet import Fernet, InvalidToken


def fernet_from_key(key: str) -> Fernet:
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_secret(fernet: Fernet, plaintext: str) -> bytes:
    return fernet.encrypt(plaintext.encode())


def decrypt_secret(fernet: Fernet, blob: bytes) -> str:
    try:
        return fernet.decrypt(blob).decode()
    except InvalidToken as e:
        raise ValueError("invalid ciphertext or wrong HOSTED_FERNET_KEY") from e
