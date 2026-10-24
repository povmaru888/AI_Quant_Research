"""Start the local fund monitoring server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8601, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        "realtime_monitor.web_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(Path(__file__).resolve().parents[1] / "src"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
