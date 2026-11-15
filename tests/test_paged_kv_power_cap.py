"""Per-die power cap of the paged allocator (pim_power_cap_w_per_die)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.core.paged_kv import PagedKVAllocator  # noqa: E402

REC = dict(page_tokens=256, page_ns=20000.0, partial_bytes=2120.0, q_bytes=1024, kv_bytes_per_token_head=256,
           pin_bandwidth_Bps=409.6e9, host_Bps=3352e9, slots=32, banks_per_device=16, n_kv_heads=2)


def _alloc(per_bank, **opts):
    """Request 7, one KV head, ``per_bank[s]`` pages on bank s (two dies of 16 banks)."""
    a = PagedKVAllocator(REC, opts)
    banks = [(s, None) for s, n in enumerate(per_bank) for _ in range(n)]
    a.inst[(7, 0)] = (0, banks)
    for s, n in enumerate(per_bank):
        a.load[s] = n
    a.pages = len(banks)
    return a


def test_off_by_default():
    a = _alloc([2] * 32)
    a.cost_ns([7])
    assert a.last_cost["compute"] == 2 * REC["page_ns"]


def test_cap_stretches_a_busy_die():
    # 16 banks x 2 pages on die 0; 2 uJ per page at a 1 W dynamic budget = 2 us per page -> 32 pages = 64 us > 40 us
    a = _alloc([2] * 16 + [0] * 16, pim_power_cap_w_per_die=1.5, pim_page_energy_j=2e-6, pim_static_w_per_die=0.5)
    a.cost_ns([7])
    assert a.last_cost["compute"] == pytest.approx(64000.0)
    assert a.last_cost["compute_uncapped"] == 2 * REC["page_ns"]


def test_cap_does_not_bind_on_a_sparse_die():
    # one busy bank: 4 pages x 2 us = 8 us of energy, timing 80 us
    a = _alloc([4] + [0] * 31, pim_power_cap_w_per_die=1.5, pim_page_energy_j=2e-6, pim_static_w_per_die=0.5)
    a.cost_ns([7])
    assert a.last_cost["compute"] == 4 * REC["page_ns"]


def test_static_above_cap_rejected():
    with pytest.raises(ValueError):
        PagedKVAllocator(REC, dict(pim_power_cap_w_per_die=0.4, pim_page_energy_j=1e-6, pim_static_w_per_die=0.5))
