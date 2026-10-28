import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from data.dataset import load_scene_splits
from model.cohere_fusion import (
    ARCHITECTURE, CohereFusion, RSFR, SEWB, SCR, frequency_bands, reliability,
)
from scripts.checkpoint import fusion_state, load_checkpoint
from scripts.losses import CohereFusionLoss


class FrozenTextStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 512)

    def forward(self, ids):
        tokens = F.normalize(self.embedding(ids.remainder(16)), dim=-1)
        valid = ids.ne(0)
        pooled = F.normalize((tokens * valid[:, :, None]).sum(1), dim=-1)
        return tokens, pooled, valid


class FrozenSemanticStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 2048, 1)

    def forward(self, visible, infrared):
        return tuple(self.projection(F.adaptive_avg_pool2d(x, (4, 4))) for x in (visible, infrared))


class PaperArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(3407)

    def test_frequency_bands_reconstruct_and_reliability_means_exposure(self):
        x = torch.rand(2, 1, 24, 32)
        torch.testing.assert_close(frequency_bands(x).sum(1), x)
        exposed = torch.ones(1, 1, 24, 24)
        mid = exposed * .5
        dark = exposed * 0
        rv_exposed, _ = reliability(exposed, mid)
        rv_mid, _ = reliability(mid, mid)
        rv_dark, _ = reliability(dark, mid)
        self.assertLess(rv_exposed[..., 6:-6, 6:-6].mean(), rv_mid[..., 6:-6, 6:-6].mean())
        self.assertEqual(rv_dark.sum().item(), 0.)

    def test_rsfr_independent_heads_receive_selected_bands(self):
        router = RSFR()
        seen = []
        handles = [head.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach()))
                   for head in router.band_residual_heads]
        x, y = torch.rand(2, 1, 16, 24), torch.rand(2, 1, 16, 24)
        output, aux = router(x, y, x, y, torch.rand(2, 512), torch.rand(2, 512))
        for handle in handles:
            handle.remove()
        self.assertEqual(len(seen), 3)
        for index in range(3):
            torch.testing.assert_close(seen[index], aux['selected_bands'][:, index])
        torch.testing.assert_close(aux['rsfr_weights'].sum(2), torch.ones(2, 3, 16, 24))
        # Zero-initialized residuals must reduce to the selected band sum.
        torch.testing.assert_close(output, aux['selected_bands'].sum(1))
        self.assertGreater((aux['rsfr_weights'] - aux['counter_rsfr_weights']).abs().sum(), 0)

    def test_sewb_padding_tokens_do_not_change_context(self):
        module = SEWB(8)
        query = torch.rand(2, 8, 40, 40)  # exercises pooled high-resolution queries
        tokens = torch.rand(2, 5, 512)
        valid = torch.tensor([[True, True, False, False, False]] * 2)
        changed = tokens.clone(); changed[:, 2:] = 1000
        torch.testing.assert_close(module.ground_text(query, tokens, valid),
                                   module.ground_text(query, changed, valid))

    def test_scr_identity_limit_and_normalized_source_weights(self):
        scr = SCR()
        with torch.no_grad():
            scr.contrast_strength.fill_(-100); scr.detail_strength.fill_(-100)
        x = torch.rand(2, 1, 16, 24)
        output, aux = scr(x, torch.rand_like(x), torch.rand_like(x), torch.rand(2, 512))
        torch.testing.assert_close(output, x)
        torch.testing.assert_close(aux['source_detail_weights'].sum(1), torch.ones(2, 16, 24))

    def test_joint_training_and_strict_checkpoint_identity(self):
        model = CohereFusion(None, dim=8, semantic_teacher=FrozenSemanticStub(), text_encoder=FrozenTextStub())
        model.train()
        self.assertFalse(model.text_encoder.training)
        self.assertFalse(model.semantic_teacher.training)
        visible, infrared = torch.rand(2, 3, 24, 32), torch.rand(2, 3, 24, 32)
        ids, replacement = torch.tensor([[1, 2, 0], [1, 3, 0]]), torch.tensor([[1, 5, 0], [1, 6, 0]])
        objective = CohereFusionLoss(bins=8)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        for _ in range(2):
            optimizer.zero_grad()
            output, aux = model(visible, infrared, ids, counter_text=replacement, return_aux=True)
            self.assertEqual(output.shape, visible.shape)
            self.assertTrue(torch.isfinite(output).all())
            loss, parts = objective(visible, infrared, visible, infrared, output, aux, ('low_light', 'ir_noise'))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name in ('gewb.output_y.weight', 'sewb.output_y.weight', 'rsfr.route_heads.0.weight',
                         'rsfr.band_residual_heads.0.weight', 'scr.contrast_head.weight'):
                gradient = dict(model.named_parameters())[name].grad
                self.assertIsNotNone(gradient, name)
                self.assertTrue(torch.isfinite(gradient).all(), name)
                self.assertGreater(gradient.abs().sum(), 0, name)
            self.assertTrue(all(p.grad is None for p in model.text_encoder.parameters()))
            self.assertTrue(all(p.grad is None for p in model.semantic_teacher.parameters()))
            optimizer.step()
        checkpoint = {'architecture': ARCHITECTURE, 'model': fusion_state(model)}
        self.assertTrue(all(not k.startswith(('text_encoder.', 'semantic_teacher.')) for k in checkpoint['model']))
        load_checkpoint(model, checkpoint)
        with self.assertRaisesRegex(ValueError, 'Expected CohereFusion'):
            load_checkpoint(model, {'model': {}})
        with self.assertRaisesRegex(ValueError, 'multiples of 8'):
            model(visible[..., :-1], infrared[..., :-1], ids)

    def test_text_targets_follow_paper_modality_order(self):
        good = torch.tensor([[.001, .001, .001, .332333, .332333, .332334],
                             [.332333, .332333, .332334, .001, .001, .001]])
        tasks = ('low_light', 'ir_noise')
        self.assertLess(CohereFusionLoss.text_routing([good], tasks),
                        CohereFusionLoss.text_routing([good.flip(1)], tasks))

    def test_manifest_rejects_cross_task_scene_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'splits.json'
            path.write_text(json.dumps({'train': ['same.png'], 'val': ['same.jpg'], 'test': ['test']}))
            with self.assertRaisesRegex(ValueError, 'leakage'):
                load_scene_splits(path)

    def test_real_training_epoch_and_checkpoint_roundtrip(self):
        from torch.utils.data import DataLoader
        from train import epoch_pass, save_checkpoint
        from argparse import Namespace
        model = CohereFusion(None, dim=8, semantic_teacher=FrozenSemanticStub(), text_encoder=FrozenTextStub())
        image = torch.rand(3, 24, 24)
        batch = [(image, image.flip(-1), image, image.flip(-1), 'low_light',
                  'Fuse infrared and visible images in low light.', 'scene_a')]
        def collate(items):
            sample = items[0]
            return (*(x[None] for x in sample[:4]), (sample[4],), (sample[5],), (sample[6],))
        class DatasetStub(torch.utils.data.Dataset):
            def __len__(self): return 1
            def __getitem__(self, index): return batch[index]
            def set_epoch(self, epoch): self.epoch = epoch
        loader = DataLoader(DatasetStub(), batch_size=1, collate_fn=collate)
        objective = CohereFusionLoss(bins=8)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        stats = epoch_pass(model, loader, objective, torch.device('cpu'), 0, optimizer, scheduler)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in stats.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paper.pth'
            save_checkpoint(path, model, optimizer, scheduler, 0, stats['loss'], Namespace(dim=8))
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            load_checkpoint(model, checkpoint)
            self.assertEqual(checkpoint['epoch'], 0)


if __name__ == '__main__':
    unittest.main()
