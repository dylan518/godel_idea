"""CLI to mint invite tokens (requires ``HOSTED_DATABASE_URL`` and ``HOSTED_ENCRYPTION_KEY``)."""

from __future__ import annotations

import argparse
import asyncio
import uuid

from EvoScientist.hosted.db import init_db, session_scope
from EvoScientist.hosted.orm_models import InviteToken


async def _create_invite(uses: int, label: str | None) -> str:
    await init_db()
    token = uuid.uuid4().hex
    async with session_scope() as session:
        session.add(
            InviteToken(token=token, uses_remaining=uses, label=label),
        )
        await session.commit()
    return token


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EvoScientist hosted administration",
        epilog="Requires HOSTED_DATABASE_URL and HOSTED_ENCRYPTION_KEY (same as API).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create-invite", help="Print a new invite token")
    p_create.add_argument("--uses", type=int, default=10)
    p_create.add_argument("--label", default=None)

    args = parser.parse_args()
    if args.cmd == "create-invite":
        tok = asyncio.run(_create_invite(args.uses, args.label))
        print(tok)


if __name__ == "__main__":
    main()
