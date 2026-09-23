import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serving.core import trace_generator as tg


def perf_db():
    return {
        "hardware": "PIM_TEST", "model": "test", "variant": "int8",
        "meta": {"require_full_layer_coverage": True},
        "architecture": {
            "sequence": {
                "prologue": ["embedding"], "pre_attn": ["attention"],
                "mlp_dense": ["act_fn"], "mlp_moe": ["moe"], "head": ["sampler"],
            },
            "catalog": {
                "dense": {"embedding": {}, "act_fn": {}},
                "attention": {"attention": {}}, "moe": {"moe": {}},
                "per_sequence": {"sampler": {"tp_stable": True}},
            },
        },
        "available_tps": [1],
        "tables": {1: {
            "dense": {
                "embedding": {"keys": [1, 32], "values": [10, 100]},
                "act_fn": {"keys": [1, 32], "values": [10, 100]},
            },
            "per_sequence": {"sampler": {"keys": [1], "values": [10]}},
            "attention": {"slices": {(0, 1): {
                "rows": [{"keys": [1, 32], "values": [10, 100]}]}}},
        }},
    }


class LayerPreflightTest(unittest.TestCase):
    def test_lazy_csv_loader_rejects_bad_prices_before_one_ns_clipping(self):
        for price in ("0", "-1", "nan", "inf"):
            with self.subTest(price=price), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp) / "tp1"
                folder.mkdir()
                (folder / "dense.csv").write_text(
                    f"layer,tokens,time_us\nembedding,1,{price}\n")
                db = perf_db()
                db["tables"] = {}
                db["root"] = tmp
                with self.assertRaisesRegex(ValueError, "Non-finite or nonpositive time_us"):
                    tg._tp_tables(db, 1)

    def test_positive_sub_nanosecond_price_can_round_to_one_ns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dense.csv"
            path.write_text("layer,tokens,time_us\nembedding,1,0.0001\n")
            df = tg._read_category_csv(path, None, require_positive=True)
            self.assertEqual(df["latency_ns"].tolist(), [1])

    def test_dense_sequence_does_not_require_inactive_moe(self):
        tg.preflight_layer_coverage(perf_db(), 1, "PIM_TEST")

    def test_missing_layer_and_unknown_catalog_entry_are_reported(self):
        db = perf_db()
        del db["tables"][1]["dense"]["embedding"]
        db["architecture"]["sequence"]["head"].append("unknown")
        with self.assertRaisesRegex(ValueError, "(?s)embedding: missing.*unknown: missing"):
            tg.preflight_layer_coverage(db, 1, "PIM_TEST")

    def test_empty_attention_is_rejected(self):
        db = perf_db()
        db["tables"][1]["attention"] = {"slices": {}}
        with self.assertRaisesRegex(ValueError, "attention: missing"):
            tg.preflight_layer_coverage(db, 1, "PIM_TEST")

    def test_bad_prices_are_rejected_for_each_category(self):
        for category, name in (("dense", "embedding"), ("per_sequence", "sampler"),
                               ("attention", "attention")):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(category=category, value=value):
                    db = perf_db()
                    table = db["tables"][1][category]
                    row = (table["slices"][(0, 1)]["rows"][0]
                           if category == "attention" else table[name])
                    row["values"][0] = value
                    with self.assertRaisesRegex(ValueError, f"{name}: empty, non-finite or nonpositive"):
                        tg.preflight_layer_coverage(db, 1, "PIM_TEST")

    def test_tp_stable_uses_the_actual_tp1_lookup(self):
        db = perf_db()
        db["available_tps"].append(2)
        db["tables"][2] = copy.deepcopy(db["tables"][1])
        del db["tables"][1]["per_sequence"]["sampler"]
        with self.assertRaisesRegex(ValueError, "sampler: missing per_sequence profile at tp=1"):
            tg.preflight_layer_coverage(db, 2, "PIM_TEST")

    def test_moe_requires_experts_at_tp1(self):
        db = perf_db()
        with self.assertRaisesRegex(ValueError, "moe: missing"):
            tg.preflight_layer_coverage(db, 1, "PIM_TEST", is_moe=True)
        db["tables"][1]["moe"] = {"rows": [{"keys": [1], "values": [10]}]}
        del db["tables"][1]["dense"]["act_fn"]
        tg.preflight_layer_coverage(db, 1, "PIM_TEST", is_moe=True)

    def test_trace_context_rejects_incomplete_export_before_emission(self):
        db = perf_db()
        del db["tables"][1]["dense"]["embedding"]
        config = {"model_type": "llama", "hidden_size": 64, "num_attention_heads": 1}
        with patch.object(tg, "_load_perf_db", return_value=db):
            with self.assertRaisesRegex(ValueError, "embedding: missing"):
                tg._build_trace_ctx(
                    "PIM_TEST", "test", config, 1, 1, 1, 1, 0, 1,
                    {}, None, False, None, None, None, "int8")


if __name__ == "__main__":
    unittest.main()
