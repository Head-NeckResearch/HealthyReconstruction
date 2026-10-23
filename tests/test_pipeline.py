"""Numerical contracts on artificial arrays; no acquired data required."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from healthy_reconstruction.preprocessing import anchored_representation
from healthy_reconstruction.geometry import kabsch, world_to_dental
from healthy_reconstruction.data import grouped_participants, read_manifest, trimmed_mean
from healthy_reconstruction.inference import maps_from_filtered, prediction_variants, common_scoring_mask, METHODS
from healthy_reconstruction.settings import default_config


class PipelineContracts(unittest.TestCase):
    def test_neutral_is_actual_raw_frame_and_gap_not_target(self):
        frames=np.zeros((5,2,3)); frames[:,:,0]=np.arange(5)[:,None]
        frames[2,0]=np.nan
        rep=anchored_representation(frames,0,1,5,n_frames=7,max_interp_gap_frames=2)
        np.testing.assert_array_equal(rep['N'],frames[0])
        self.assertTrue(np.any(rep['X'][0]!=0))
        self.assertTrue(rep['B_imputed'].any())
        self.assertFalse((rep['B_imputed']&rep['B_target']).any())
        self.assertFalse(np.shares_memory(rep['N'],frames))

    def test_rigid_fit_and_dental_frame(self):
        points=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],[1.,1.,3.]])
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        target=points@rotation.T+np.array([5.,-4.,3.])
        fitted,translation=kabsch(points,target)
        np.testing.assert_allclose(points@fitted.T+translation,target,atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(fitted),1.)
        dental,ok=world_to_dental(np.stack([points,target]))
        self.assertTrue(ok.all())
        np.testing.assert_allclose(dental[0],dental[1],atol=1e-12)

    def test_participant_split_is_disjoint_and_repeatable(self):
        tokens=[f'{i:032x}' for i in range(8)]
        train,val=grouped_participants(tokens,13,.25)
        self.assertFalse(set(train)&set(val)); self.assertEqual(set(train+val),set(tokens))
        self.assertEqual((train,val),grouped_participants(tokens[::-1],13,.25))

    def test_manifest_rejects_role_leakage_and_clinical_fields(self):
        def record(number,role):
            return {'sample_id':f'{number:032x}','participant_id':f'{1:032x}','role':role,'task':'M1',
                    'visit_index':1,'file':f'{number}.npz','neutral_raw_index':0}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest.json'
            manifest={'schema_version':1,'coordinate_system':'dental','records':[record(1,'optimizer'),record(2,'validation')]}
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Participant overlap'):
                read_manifest(path)
            manifest['records'][1]['participant_id']=f'{2:032x}'
            manifest['records'][0]['clinical_note']='not accepted'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'clinical metadata'):
                read_manifest(path)

    def test_maps_keep_own_neutral_and_direction_missing(self):
        observed=np.array([[[2.,0.,0.]],[[.1,0.,0.]]])
        reference=np.array([[[-3.,0.,0.]],[[-.1,0.,0.]]])
        maps=maps_from_filtered(observed,reference,np.zeros((1,3)),np.zeros((1,3)),np.ones((2,1),bool),1.)
        self.assertAlmostEqual(maps['counter_direction'][0,0],2.)
        self.assertTrue(np.isnan(maps['counter_direction'][1,0]))
        self.assertAlmostEqual(maps['reduced_movement'][0,0],1.)
        shifted=maps_from_filtered(observed+7,reference+9,np.full((1,3),7.),np.full((1,3),9.),np.ones((2,1),bool),1.)
        np.testing.assert_allclose(maps['counter_direction'],shifted['counter_direction'],equal_nan=True)

    def test_final_variants_are_unblended_and_common_support(self):
        neutral=np.ones((2,3)); static=neutral*2
        motion=np.ones((3,2,3))*3; observed_motion=motion*2
        nr=neutral*4; xr=motion*3
        variants=prediction_variants(neutral,static,motion,observed_motion,nr,xr)
        self.assertEqual(tuple(variants),METHODS)
        np.testing.assert_array_equal(variants['ae'][0],static[None]+motion)
        np.testing.assert_array_equal(variants['without_static_projection'][0],neutral[None]+motion)
        variants['healthy_mean_trajectory'][0][0,0]=np.nan
        support=common_scoring_mask(neutral[None]+observed_motion,np.ones((3,2),bool),variants)
        self.assertFalse(support[0,0]); self.assertTrue(support[1:,0].all())

    def test_public_config_has_no_blends_or_temporal_loss(self):
        c=default_config()
        self.assertEqual(c['reconstruction'],{'neutral_identity_blend':0.,'motion_anchor_strength':0.})
        self.assertEqual(c['training']['dynamic']['lambda_velocity'],0)
        self.assertEqual(c['training']['dynamic']['lambda_acceleration'],0)
        self.assertEqual(c['training']['static']['hidden'],256)
        self.assertEqual(c['training']['dynamic']['latent'],40)

    def test_masked_nan_loss_and_optimizer_only_normalizer(self):
        import torch
        from healthy_reconstruction.models import masked_error_sums, compute_norm
        prediction=torch.tensor([[[1.,2.,3.],[float('nan')]*3]],requires_grad=True)
        target=torch.tensor([[[0.,0.,0.],[float('nan')]*3]])
        numerator,denominator=masked_error_sums(prediction,target,torch.tensor([[1.,0.]]))
        loss=numerator/denominator; loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(prediction.grad).all())
        values=np.array([[[1.,2.,3.]],[[1000.,2000.,3000.]]])
        mean,std=compute_norm(values[:1],np.ones((1,1),bool))
        np.testing.assert_array_equal(mean,[1.,2.,3.]); np.testing.assert_array_equal(std,[1.,1.,1.])


if __name__=='__main__':
    unittest.main()
