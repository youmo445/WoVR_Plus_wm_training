import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from wovrplus.config import ROOT, resolve_config
from wovrplus.data.registry import SampleAdapter


class PublicConfigTests(unittest.TestCase):
    def test_online_cached_recipes_share_training_and_data_plan(self):
        for method, family in [('raymap_vace', 'raymap_vace'), ('motionimage', 'motionimage'),
                               ('vector_adaln', 'vector')]:
            online = resolve_config([ROOT/f'configs/train/adjust_bottle_{method}_100k.yaml'])
            cached = resolve_config([ROOT/f'configs/train/adjust_bottle_{method}_cached_100k.yaml'])
            self.assertEqual(online['model'], cached['model'])
            self.assertEqual(online['training'], cached['training'])
            self.assertEqual(online['model']['family'], family)
            self.assertEqual(online['data']['cache']['mode'], 'off')
            self.assertEqual(cached['data']['cache']['mode'], 'read')
            self.assertNotIn('condition_cache', online)
            for key in ('base_path', 'kwargs', 'sampling', 'resize_to'):
                self.assertEqual(online['data'][key], cached['data'][key])
            self.assertEqual(len(online['data']['base_path']), 5)
            self.assertEqual(online['training']['max_steps'], 100000)
            self.assertEqual(online['training']['save_steps'], 5000)
            if family != 'raymap_vace':
                self.assertEqual(cached['condition_cache']['kind'], family)
            if family == 'vector':
                self.assertEqual(online['model']['action_mode'], 'modulation')

    def test_legacy_family_and_import_aliases(self):
        from wovrplus.models.action import WanActionModel
        from wovrplus.models.vector import WanVectorModel
        from wovrplus.models.physcanvas import WanPhysCanvasModel
        from wovrplus.models.motionimage import WanMotionImageModel
        from wovrplus.models.vace import VaceWanModel
        from wovrplus.models.raymap_vace import RaymapVACEModel
        self.assertIs(WanActionModel, WanVectorModel)
        self.assertIs(WanPhysCanvasModel, WanMotionImageModel)
        self.assertIs(VaceWanModel, RaymapVACEModel)
        for old, new, recipe in [('wan_vace', 'raymap_vace', 'raymap_vace'),
                                 ('wan_action', 'vector', 'vector_adaln'),
                                 ('wan_physcanvas', 'motionimage', 'motionimage')]:
            cfg = resolve_config([ROOT/f'configs/train/adjust_bottle_{recipe}_100k.yaml'],
                                 [f'model.family={old}'])
            self.assertEqual(cfg['model']['family'], new)

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
                from wovrplus.models.vector import normalize_action_mode
                self.assertEqual(cfg['model']['action_mode'], normalize_action_mode(mode))

    def test_online_mesh_adapter_uses_frame_aligned_renderer(self):
        from PIL import Image
        cfg = resolve_config([ROOT/'configs/train/robotwin_motionimage.yaml'])
        sample = dict(video=[Image.new('RGB',(32,96)) for _ in range(13)],
                      meta=dict(path='example.hdf5', frame_ids=list(range(13)),
                                camera_names=['head_camera','left_camera','right_camera']))
        with patch('wovrplus.data.motionimage_renderer.RoboTwinMeshRenderer') as renderer:
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
