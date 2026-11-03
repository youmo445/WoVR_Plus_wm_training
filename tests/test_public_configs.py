import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from wovrplus.config import ROOT, resolve_config
from wovrplus.data.registry import SampleAdapter


class PublicConfigTests(unittest.TestCase):
    def test_all_recipes_resolve_under_external_asset_root(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, WOVR_ASSET_ROOT=tmp):
            for path in (ROOT / 'configs/train').glob('*.yaml'):
                cfg = resolve_config([path])
                self.assertTrue(Path(cfg['model']['model_dir']).is_relative_to(Path(tmp)))
                paths = cfg['data']['base_path']
                for item in paths if isinstance(paths, list) else [paths]:
                    self.assertTrue(Path(item).is_relative_to(Path(tmp)))
                self.assertIsNone(cfg['model']['checkpoint'])
            for mode in ('cross_attn', 'adaln', 'cross_attn_and_adaln'):
                cfg = resolve_config([ROOT/'configs/train/adjust_bottle_vector_cached_100k.yaml'],
                                     [f'model.action_mode={mode}'])
                from wovrplus.models.action import normalize_action_mode
                self.assertEqual(cfg['model']['action_mode'], normalize_action_mode(mode))

    def test_online_mesh_adapter_uses_frame_aligned_renderer(self):
        from PIL import Image
        cfg = resolve_config([ROOT/'configs/train/robotwin_physcanvas.yaml'])
        sample = dict(video=[Image.new('RGB',(32,96)) for _ in range(13)],
                      meta=dict(path='example.hdf5', frame_ids=list(range(13)),
                                camera_names=['head_camera','left_camera','right_camera']))
        with patch('wovrplus.data.physcanvas_renderer.RoboTwinMeshRenderer') as renderer:
            renderer.return_value.render.return_value = sample['video']
            adapter = SampleAdapter([sample], cfg)
            result = adapter[0]
            self.assertEqual(len(result['motion_video']), len(result['video']))
            renderer.return_value.render.assert_called_once_with(sample['meta'])

    def test_projection_module_imports_without_legacy_adapters(self):
        from wovrplus.data.projection import get_transformation_matrix_from_quat
        pose = np.array([[1.,2.,3.,0.,0.,0.,1.]])
        transform = get_transformation_matrix_from_quat(pose)
        np.testing.assert_allclose(transform[0,:3,3], pose[0,:3])
        np.testing.assert_allclose(transform[0,:3,:3], np.eye(3))


if __name__ == '__main__':
    unittest.main()
