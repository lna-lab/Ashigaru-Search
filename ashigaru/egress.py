"""Egress sovereignty — a default-deny gate + audit trail for outbound connections.

A local "sovereign" search agent must not be turnable against its own machine. Without a
gate, ``fetch_url`` will GET *any* URL the model produces (or any host a fetched page
redirects to) and feed the response back into the LLM — an SSRF vector straight at the
loopback/LAN services this box runs (vLLM, the brain, the journal, cloud metadata at
169.254.169.254, …). The :class:`EgressGate` closes that, and every decision is audited.

The doctrine, mirrored from the wider Lna-Lab design: **references in, secrets out — but no
audit ⇒ no act.** The gate audits BEFORE it acts; if the audit write itself fails, the gate
DENIES (a connection may never go out unrecorded). The audit record carries host + reason
only — never the path/query — so the log can't leak what was being looked up.

Policy (per :meth:`EgressGate.check`, by ``mode``):

* a **private / link-local / reserved IP** (RFC1918, 169.254.x metadata, …) is NEVER allowed,
  in any mode — that is the SSRF block;
* ``mode="search"`` (and infra like the reader proxy): loopback + the configured allow-list;
* ``mode="fetch"``: a *public* host that a prior ``web_search`` surfaced (the search result is
  the capability) or is allow-listed — loopback/private are refused, so the model can't fetch
  the box itself. With no SourceRegistry (legacy URL mode) any public host is allowed;
* ``mode="redirect"`` (a hop while following a fetch): any *public* host (a real result may
  redirect to a CDN / www host) but never loopback/private.
"""
from __future__ import annotations

import ipaddress
import json
import os
import time
from urllib.parse import urlsplit


class EgressDenied(Exception):
    """Raised when the egress policy refuses an outbound connection."""


class AuditWriteError(Exception):
    """Raised when an audit record cannot be persisted — the gate then fails closed."""


class NullAudit:
    """Audit sink that records nowhere and never fails (policy still enforced, just not logged).
    Used when no audit path is configured: ``no-audit ⇒ no-act`` applies to audit *failure*,
    not to deliberately running without a persistent log."""

    def record(self, **_fields) -> None:  # noqa: D401 - trivial
        return None


class FileAuditLog:
    """Append-only, crash-safe JSONL audit trail (one fsync'd line per decision).

    Mirrors the 関所's ledger append: O_APPEND + a newline-repair so a torn write can't corrupt
    the next record. A failed write raises :class:`AuditWriteError` so the caller (the gate)
    can fail closed."""

    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)

    def record(self, **fields) -> None:
        rec = {"t": time.time(), **fields}
        try:
            line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                size = os.fstat(fd).st_size
                if size > 0:
                    os.lseek(fd, size - 1, os.SEEK_SET)
                    if os.read(fd, 1) != b"\n":
                        os.write(fd, b"\n")
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as e:
            raise AuditWriteError(f"egress audit write failed ({self.path}): {e}") from e


def _host_of(url: str) -> str:
    """Lower-cased hostname of ``url`` (empty string if unparsable). A scheme-less ``host:port``
    is handled by retrying with a synthetic scheme so userinfo/port are still stripped."""
    parts = urlsplit(url)
    host = parts.hostname
    if host is None:
        host = urlsplit("//" + url).hostname
    return (host or "").lower()


def _classify(host: str) -> str:
    """Classify ``host`` as ``loopback`` | ``private`` | ``public`` | ``unparsable``.

    ``localhost`` and any IP in ``127.0.0.0/8`` / ``::1`` are loopback. A non-globally-routable
    IP (RFC1918, link-local 169.254.x metadata, reserved, …) is ``private`` and always blocked.
    A real hostname can't be resolved here, so it's treated as ``public`` (DNS-rebinding to a
    private IP is a documented out-of-scope limitation — matches the upstream design). The
    classic ``127.evil.com`` trap is a hostname, not an IP, so it is ``public``, not loopback."""
    if not host:
        return "unparsable"
    h = host.strip().lower()
    if h == "localhost":
        return "loopback"
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return "public"          # a hostname, not an IP literal
    if ip.is_loopback:
        return "loopback"
    if not ip.is_global:
        return "private"         # RFC1918 / link-local / reserved / multicast / unspecified
    return "public"


class EgressGate:
    """Default-deny outbound policy, audited per decision (see module docstring)."""

    def __init__(self, audit=None, allow_hosts=(), *, fetch_open: bool = True):
        self._audit = audit or NullAudit()
        self._allow = {h.strip().lower() for h in allow_hosts if h and h.strip()}
        self._fetch_open = fetch_open

    def check(self, url: str, *, mode: str = "search", discovered_hosts=()) -> str:
        """Permit an outbound request to ``url`` or raise :class:`EgressDenied`.

        Returns the lower-cased host on success. Audits the decision first; an
        :class:`AuditWriteError` propagates (fail closed). ``discovered_hosts`` are the hosts a
        prior ``web_search`` surfaced (the fetch capability)."""
        host = _host_of(url)
        cls = _classify(host)
        discovered = {h.strip().lower() for h in discovered_hosts if h and h.strip()}

        allowed, reason = False, "default-deny"
        if cls == "unparsable":
            reason = "unparsable-host"
        elif cls == "private":
            reason = "private-ip"                       # SSRF block — never allowed
        elif host in self._allow:
            allowed, reason = True, "allow-list"
        elif cls == "loopback":
            if mode in ("search", "infra"):
                allowed, reason = True, "loopback"      # our own SearXNG / reader proxy
            else:
                reason = "loopback-in-fetch"            # a fetch must never target this box
        else:  # public host
            if mode == "redirect":
                allowed, reason = True, "redirect-public"
            elif mode == "fetch":
                if not self._fetch_open:
                    reason = "fetch-closed"
                elif not discovered or host in discovered:
                    # no registry (legacy) -> any public ok; with a registry -> must be surfaced
                    allowed, reason = True, ("fetch-open" if not discovered else "fetch-discovered")
                else:
                    reason = "not-discovered"
            elif mode == "search":
                reason = "search-nonloopback"           # remote SearXNG must be allow-listed

        # Audit BEFORE acting; host + reason only (no path/query → no secrets).
        self._audit.record(actor="egress", action="egress.check",
                           target=host or "<unparsable>", outcome="allowed" if allowed else "denied",
                           detail=f"mode={mode} reason={reason}")
        if not allowed:
            raise EgressDenied(f"outbound to {host or url!r} denied (mode={mode}, reason={reason})")
        return host
