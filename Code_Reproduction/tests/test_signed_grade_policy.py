import unittest
from types import SimpleNamespace
import numpy as np
from src import trust,recovery
from src.count_feedback import feedback
from src.signed_grade_policy import select_round,requester_round

class SharedPolicyTests(unittest.TestCase):
    def test_selection_ignores_alias_order(self):
        rng=np.random.default_rng(91);n=12;D=20
        packet={k:rng.random((n,D))<.4 for k in ('bits','hi','nonzero')}
        r=np.asarray([.95]*3+[.55]*9);active=np.ones(n,bool);tasks=np.arange(10)
        a=select_round(packet,active,r,tasks,450)
        permutation=rng.permutation(n)
        b=select_round({k:v[permutation] for k,v in packet.items()},active[permutation],r[permutation],tasks,450)
        self.assertEqual(set(a['members']),set(permutation[b['members']]))
        self.assertEqual(a['cost_numerator'],b['cost_numerator'])
        self.assertEqual(a['active_packet_tie_collisions'],0)
    def test_default_feedback_matches_existing_frozen_rule(self):
        rng=np.random.default_rng(9);public=rng.random((100,30));head=recovery.initialize(public,10.,80.)
        raw=np.asarray([[25,26,27,28,29,25,26,27,28,29],[25.1,26.1,27.1,28.1,29.1,25.1,26.1,27.1,28.1,29.1],[25.2,26.2,27.2,28.2,29.2,25.2,26.2,27.2,28.2,29.2]],dtype=np.float32).astype(float)
        raw=np.tile(raw,(1,10))
        r=np.asarray([.95,.96,.5]);direct=trust.feedback(raw,r)
        anchor=recovery.decode_actual(direct['credible_consensus'],head)
        secondary,_=recovery.complete(direct['credible_consensus'],anchor,head)
        secondary[direct['has_credible']]=np.nan
        old=feedback(raw,r,secondary,trust.Config(),head['offset'],head['scale'],.5)
        new=requester_round(raw,r,head)
        np.testing.assert_array_equal(old['next_reputation'],new['next_reputation'])
        np.testing.assert_array_equal(old['effective_evidence'],new['effective_evidence'])
        shuffled=requester_round(raw[[2,0,1]],r[[2,0,1]],head)
        np.testing.assert_array_equal(new['next_reputation'][[2,0,1]],shuffled['next_reputation'])
if __name__=='__main__':unittest.main()
