"""Off-die softmax term of the paged allocator (score_roundtrip_bytes / offdie_bandwidth_Bps in the paged record)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.core.paged_kv import PagedKVAllocator  # noqa: E402

REC = dict(page_tokens=256, page_ns=20800.0, partial_bytes=2120.0, q_bytes=1024, kv_bytes_per_token_head=256,
           pin_bandwidth_Bps=409.6e9, host_Bps=3352e9, slots=32, banks_per_device=16, n_kv_heads=2)


def _alloc(rec, pages_per_bank=3, **opts):
    a = PagedKVAllocator(rec, opts)
    # one request, two KV-head instances on device 0, pages spread over its 16 banks
    for h in range(2):
        banks = [((h * 16 + i) % 16, 0) for i in range(pages_per_bank * 8)]
        a.inst[(7, h)] = (0, banks)
        for s, _ in banks:
            a.load[s] += 1
        a.pages += len(banks)
    a.dev_load[0] = a.pages
    return a


def _base(rec, a, pages):
    pins = (pages * rec["partial_bytes"] + 2 * rec["q_bytes"] + 2 * rec["kv_bytes_per_token_head"]) / 409.6
    return pins, max(a.load) * rec["page_ns"], pages * rec["partial_bytes"] / 3352


def test_default_bundle_unchanged():
    a = _alloc(REC)
    pins, compute, host = _base(REC, a, a.pages)
    assert abs(a.cost_ns([7]) - (compute + pins + host)) < 1e-6


def test_roundtrip_on_shared_pins():
    rec = dict(REC, score_roundtrip_bytes=8192)
    a = _alloc(rec)
    pins, compute, host = _base(rec, a, a.pages)
    assert abs(a.cost_ns([7]) - (compute + pins + a.pages * 8192 / 409.6 + host)) < 1e-6


def test_roundtrip_overlapped_tail():
    rec = dict(REC, score_roundtrip_bytes=8192)
    a = _alloc(rec, pin_overlap=True, pin_overlap_compute_factor=1.0)
    pins, compute, host = _base(rec, a, a.pages)
    pins += a.pages * 8192 / 409.6
    head = min(2, a.pages, 32) * 1024 / 409.6
    tail = min(a.pages, 32) * (2120 + 8192) / 409.6
    assert abs(a.cost_ns([7]) - (max(head + compute + tail, pins) + host)) < 1e-6


def test_roundtrip_point_to_point_die_side():
    rec = dict(REC, score_roundtrip_bytes=8192, offdie_bandwidth_Bps=819.2e9)
    a = _alloc(rec)
    pins, compute, host = _base(rec, a, a.pages)
    die = (pins * 409.6 + a.pages * 8192) / 819.2
    assert abs(a.cost_ns([7]) - (compute + max(pins, die) + host)) < 1e-6
