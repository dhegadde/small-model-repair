"""Pilot 3: real 2-bit storage must reproduce the simulated (fake-quantized) experts + trained corrections."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from repair.cc_code import fake_quantize_experts_
from repair.fb_packed import PackedExperts, pack, pack_experts_, save_deltas, unpack
from repair.fb_fold import fold_experts_
from repair.fb_recover import add_expert_deltas


def toy():
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(vocab_size=300, hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                         num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[])
    return Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).eval()


class TestFbPacked(unittest.TestCase):
    def test_pack_roundtrip(self):
        for nbits in (2, 4):
            codes = torch.randint(0, 2 ** nbits, (3, 5, 64), dtype=torch.uint8)
            packed = pack(codes, nbits)
            self.assertEqual(tuple(packed.shape), (3, 5, 64 * nbits // 8))
            self.assertTrue(torch.equal(unpack(packed, nbits), codes))

    def test_packed_matches_simulated(self):
        base = toy()
        sim = copy.deepcopy(base)
        fake_quantize_experts_(sim, 2)
        add_expert_deltas(sim, 4)
        torch.manual_seed(1)
        for name, p in sim.named_parameters():  # trained-looking corrections (A starts at zero), as large as W itself
            if name.endswith(".A"):
                p.data.normal_(0, 0.05)
        with tempfile.TemporaryDirectory() as d:
            save_deltas(sim, Path(d) / "deltas.pt")
            deltas = torch.load(Path(d) / "deltas.pt")
        real = copy.deepcopy(base)
        self.assertEqual(pack_experts_(real, 2, deltas=deltas, device="cpu"), 2)
        self.assertTrue(all(isinstance(m.mlp.experts, PackedExperts) for m in real.model.layers))
        for layer_sim, layer_real in zip(sim.model.layers, real.model.layers):
            for pname in ("gate_up_proj", "down_proj"):
                ws = getattr(layer_sim.mlp.experts, pname).float()
                wr = layer_real.mlp.experts.weight(pname, torch.arange(8)).float()
                self.assertLess(((ws - wr).abs().mean() / ws.abs().mean()).item(), 0.005)
        plain = copy.deepcopy(base)
        pack_experts_(plain, 2, device="cpu")
        ids = torch.randint(0, 300, (2, 24), generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            a, b, c = (m(ids).logits.float() for m in (sim, real, plain))
        rel = ((a - b).abs().mean() / a.abs().mean()).item()
        self.assertLess(rel, 0.02)  # only fp16 scale/zero and bf16 corrections differ from training
        # and the corrections are really applied: without them the output differs much more
        self.assertGreater(((a - c).abs().mean() / a.abs().mean()).item(), 5 * rel)

    def test_svd_corrections_shrink_the_rounding_error(self):
        e = toy().model.layers[0].mlp.experts
        plain, svd = PackedExperts(e, 2, 64, device="cpu"), PackedExperts(e, 2, 64, device="cpu", svd_rank=4)
        self.assertTrue(svd.has_deltas)
        idx = torch.arange(8)
        for pname in ("gate_up_proj", "down_proj"):
            w = getattr(e, pname).float()
            err_plain = (w - plain.weight(pname, idx).float()).norm()
            err_svd = (w - svd.weight(pname, idx).float()).norm()
            self.assertLess(err_svd, 0.97 * err_plain)

    def test_folded_bf16_checkpoint_equals_packed_model(self):
        base = toy()
        folded, packed, svd_folded, svd_packed = (copy.deepcopy(base) for _ in range(4))
        fold_experts_(folded, 2, device="cpu")
        pack_experts_(packed, 2, device="cpu")
        fold_experts_(svd_folded, 2, svd_rank=4, device="cpu")
        pack_experts_(svd_packed, 2, device="cpu", svd_rank=4)
        ids = torch.randint(0, 300, (2, 24), generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            self.assertTrue(torch.equal(folded(ids).logits, packed(ids).logits))
            self.assertTrue(torch.equal(svd_folded(ids).logits, svd_packed(ids).logits))

    def test_storage_is_2_5_bits_per_weight(self):
        e = toy().model.layers[0].mlp.experts
        pe = PackedExperts(e, 2, 64, device="cpu")
        bits = 8 * sum(b.numel() * b.element_size() for b in pe.buffers()) / (e.gate_up_proj.numel() + e.down_proj.numel())
        self.assertAlmostEqual(bits, 2.5)


if __name__ == "__main__":
    unittest.main()
