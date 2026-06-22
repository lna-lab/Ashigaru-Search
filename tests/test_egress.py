"""Egress gate policy + audit fail-closed. These guard the SSRF defence — don't loosen
without understanding which host class you're letting through."""
import json

import pytest

from ashigaru.egress import (
    AuditWriteError,
    EgressDenied,
    EgressGate,
    FileAuditLog,
    NullAudit,
)


def _allows(gate, url, **kw):
    try:
        gate.check(url, **kw)
        return True
    except EgressDenied:
        return False


def test_private_and_metadata_always_blocked():
    g = EgressGate(NullAudit())
    # RFC1918 / link-local metadata are refused even in fetch mode with the host "discovered"
    assert not _allows(g, "http://10.0.0.5/x", mode="fetch", discovered_hosts={"10.0.0.5"})
    assert not _allows(g, "http://192.168.1.1/x", mode="fetch", discovered_hosts={"192.168.1.1"})
    assert not _allows(g, "http://169.254.169.254/latest/meta-data", mode="fetch",
                       discovered_hosts={"169.254.169.254"})


def test_loopback_search_yes_fetch_no():
    g = EgressGate(NullAudit())
    assert _allows(g, "http://localhost:8888/search", mode="search")       # our SearXNG
    assert not _allows(g, "http://localhost:8011/v1", mode="fetch")        # fetching the box = SSRF
    assert not _allows(g, "http://127.0.0.1:8011", mode="redirect")        # redirect to box = SSRF


def test_fetch_capability_is_search_results():
    g = EgressGate(NullAudit())
    disc = {"example.com"}
    assert _allows(g, "https://example.com/a", mode="fetch", discovered_hosts=disc)
    assert not _allows(g, "https://evil.com/a", mode="fetch", discovered_hosts=disc)
    # legacy (no registry -> no discovered set): any public host is allowed, private still blocked
    assert _allows(g, "https://anything.com/a", mode="fetch", discovered_hosts=())


def test_redirect_allows_public_blocks_private():
    g = EgressGate(NullAudit())
    assert _allows(g, "https://cdn.jsdelivr.net/x", mode="redirect")        # CDN hop is fine
    assert not _allows(g, "http://10.0.0.1/x", mode="redirect")             # private hop = SSRF


def test_hostname_prefix_trap_not_loopback():
    # "127.evil.com" is a hostname, not 127.0.0.0/8 — must NOT be treated as loopback
    g = EgressGate(NullAudit())
    assert _allows(g, "http://127.evil.com", mode="fetch", discovered_hosts={"127.evil.com"})


def test_allow_list():
    g = EgressGate(NullAudit(), allow_hosts=["my.searx"])
    assert _allows(g, "http://my.searx/search", mode="search")
    assert not _allows(g, "http://other.host/search", mode="search")


def test_audit_records_host_and_reason(tmp_path):
    p = str(tmp_path / "egress.jsonl")
    g = EgressGate(FileAuditLog(p))
    g.check("http://localhost:8888/search?q=secret%20query", mode="search")
    recs = [json.loads(l) for l in open(p)]
    assert recs[0]["target"] == "localhost" and recs[0]["outcome"] == "allowed"
    # the path/query (which could carry secrets) must NOT be in the audit record
    assert "secret" not in json.dumps(recs)


def test_no_audit_no_act():
    class FailingAudit:
        def record(self, **k):
            raise AuditWriteError("disk full")
    g = EgressGate(FailingAudit())
    # a request the policy WOULD allow must still fail closed if it can't be audited
    with pytest.raises(AuditWriteError):
        g.check("http://localhost:8888/search", mode="search")
