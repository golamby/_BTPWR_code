import unittest
from dataclasses import replace
import numpy as np
from src import reapp_native as reapp


class REAPPTests(unittest.TestCase):
    def test_original_score_and_asymmetric_update(self):
        raw=np.array([[100.,100.],[100.,100.],[200.,200.]])
        r=np.array([.96,.5,.5]);cfg=reapp.Config(mu=.7)
        refs=reapp.credible_reference(raw,r,cfg)
        np.testing.assert_allclose(refs,[100,100])
        nxt,count=reapp.feedback(raw,r,refs,cfg)
        np.testing.assert_allclose(nxt,[.972,.65,.05])
        np.testing.assert_array_equal(count,[2,2,2])

    def test_top_phi_and_strict_native_class_boundary(self):
        cfg=reapp.Config(phi=1)
        r=np.array([.9,.95,.99])
        np.testing.assert_array_equal(reapp.classes(r,cfg),[1,2,2])
        np.testing.assert_allclose(reapp.credible_reference(np.array([[1.],[5.],[9.]]),r,cfg),[9.])

    def test_no_reference_and_excluded_states_preserved(self):
        raw=np.array([[20.],[20.]])
        r=np.array([.5,.1])
        nxt,_=reapp.feedback(raw,r,np.array([np.nan]))
        np.testing.assert_array_equal(nxt,r)
        nxt,_=reapp.feedback(raw,r,np.array([20.]))
        self.assertEqual(nxt[1],.1)

    def test_masked_factorization_preserves_actual_observations(self):
        x=np.array([[10.,12.,np.nan],[11.,13.,np.nan],[12.,14.,16.]])
        head=reapp.MFDC(replace(reapp.Config(),rank=1,max_epochs=30),seed=1)
        completed,diag=head.complete(x)
        np.testing.assert_array_equal(completed[np.isfinite(x)],x[np.isfinite(x)])
        self.assertTrue(np.isfinite(completed).all())
        self.assertEqual(diag['observations'],7)

    def test_campaign_ignores_hidden_ground_truth(self):
        cfg=replace(reapp.Config(),recovery_start_cycle=1,rank=1,max_epochs=20)
        world=dict(presence=np.ones((2,3,3),bool),offer_mask=np.ones((2,3),bool),
                   initial_reputation=np.array([.96,.5,.5]),actual_cost=np.ones((2,3,3)),
                   public_reference=np.array([[.2,.3],[.3,.4],[.4,.5]]),
                   reports_check=np.array([[[16.,19.,22.],[16.,19.,22.],[35.,35.,35.]]]*2),
                   native_offset=10.,native_scale=30.,budget=3.)
        a=reapp.campaign(world,dict(seed=1),config=cfg)
        world['truth_native_private']=np.full((3,2),1e10)
        world['honest_private']=np.zeros(3,bool)
        b=reapp.campaign(world,dict(seed=1),config=cfg)
        np.testing.assert_array_equal(a['predictions'],b['predictions'])
        np.testing.assert_array_equal(a['reputations'],b['reputations'])
        self.assertTrue(np.isnan(a['resources'][:,0]).all())
        # One trusted route covers every region: mature native rule acquires
        # that same worker once, rather than charging each visited region.
        self.assertEqual(a['resources'][0,1],3.)
        self.assertEqual(a['resources'][0,2],1.)
        c=reapp.campaign(world,dict(seed=1),config=cfg,cap_adapter=True)
        self.assertTrue((c['resources'][:,0]<=3.).all())


if __name__=='__main__':unittest.main()
