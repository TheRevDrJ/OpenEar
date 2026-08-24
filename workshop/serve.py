#!/usr/bin/env python3
"""workshop/serve.py — OpenEar's workshop server. Read-only. Port 8002.

WHAT THIS IS
------------
The tool-bench that sits beside the app (rules.md §16): a small server whose only
job is to show me things about this project in Jonathan's own browser. Today it
serves one bench — the bug tracker. More will accrete; that is the pattern.

    python3 workshop/serve.py        ->  http://127.0.0.1:8002/

WHY PORT 8002
-------------
The convention is that a project's tools live in that project's hundred-block
(ports.md) — FirstLight runs on 6400, so its workshop is 6402. OpenEar runs on
**80**, which has no block, and the literal +2 would be port 82: below 1024, so
macOS would demand sudo every time Jonathan opened his own bug tracker. 8002
keeps the rule's intent — the tools sit in the app's range, unmistakably "port
80's block" — without the friction. His call, 2026-08-23.

WHY PURE STDLIB, NO FASTAPI
---------------------------
The app's venv lives on Zora and needs CUDA. This runs on the Mac, where
Jonathan actually reads. Depending on nothing means it works on either machine,
with no install, forever — the same reasoning as score_translation.py.

⛔⛔ READ-ONLY BY CONSTRUCTION, NOT BY A FLAG
--------------------------------------------
***His standing ruling: "I also never want to have write access to this tool.
read only." and "I WANT BUGS TO GO THROUGH BOB."***

FirstLight's workshop carries a `FILING_ENABLED = False` switch because the
filing form was built first and disabled after. Here there is no form and no
write route to disable — **the server has no code path that mutates anything.**
That is strictly stronger: a flag can be flipped by someone who does not know
why it is off; an absent route cannot.

⛔ So if a future me is asked for a "file a bug" button: that is a conversation
with Jonathan, not an edit to this file. Triage is a judgment (rules.md §20's
second counterweight) — is this a duplicate, which area, what severity — and a
judgment does not go in a form.

⛔⛔ THE TRACKER IS READ FRESH OFF DISK ON EVERY REQUEST
-------------------------------------------------------
Never cached, never imported, never held in memory between requests. FirstLight
learned this the hard way (TL-038): a cached tracker made the bench permanently
stale and Jonathan read a closed entry as open three times in one session. A
handler that opens the file when asked has no cache to go stale.

`/__bugs-stamp` is the cheap half of the same question — mtime and size, ~40
bytes — so the page can poll for "has it changed?" without pulling the whole
tracker to be told "no".
"""

import http.server
import json
import os
import socketserver
import sys
from pathlib import Path

PORT = 8002
PROJECT = Path(__file__).resolve().parent.parent
TRACKER = PROJECT / "docs" / "bugs.json"
WEB_ROOT = PROJECT / "workshop"


class Workshop(http.server.SimpleHTTPRequestHandler):
    """Static files out of workshop/, plus two read-only JSON endpoints."""

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB_ROOT), **kw)

    # ⛔ No do_POST, no do_PUT, no do_DELETE. See the module header: read-only is a
    # property of this class, not a setting. SimpleHTTPRequestHandler answers 501
    # for any verb it has no handler for, which is the correct refusal.

    def _json(self, payload: str, code: int = 200) -> None:
        body = payload.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        # ⛔ no-store, or the BROWSER caches what the server deliberately does not.
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                                    # noqa: N802
        path = self.path.split("?", 1)[0]

        if path == "/__bugs":
            try:
                self._json(TRACKER.read_text(encoding="utf-8"))
            except Exception as e:
                self._json(json.dumps({"error": str(e)}), 500)
            return

        if path == "/__bugs-stamp":
            try:
                st = TRACKER.stat()
                self._json(json.dumps({"m": st.st_mtime, "s": st.st_size}))
            except Exception as e:
                self._json(json.dumps({"error": str(e)}), 500)
            return

        if path == "/":
            self.path = "/index.html"

        super().do_GET()

    def log_message(self, fmt, *args):
        # Quiet by default — a bench that spams the terminal gets closed.
        if os.environ.get("WORKSHOP_VERBOSE"):
            super().log_message(fmt, *args)


def main() -> None:
    if not TRACKER.exists():
        print(f"WARNING: no tracker at {TRACKER} — the bugs bench will show an error.")
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("127.0.0.1", PORT), Workshop) as httpd:
            print(f"OpenEar workshop  ->  http://127.0.0.1:{PORT}/")
            print(f"  tracker: {TRACKER}")
            print("  read-only. Ctrl-C to stop.")
            httpd.serve_forever()
    except OSError as e:
        sys.exit(f"ERROR: cannot bind port {PORT} — {e}\n"
                 f"       Something else is on it, or a previous workshop is still running.")
    except KeyboardInterrupt:
        print("\nworkshop stopped.")


if __name__ == "__main__":
    main()
