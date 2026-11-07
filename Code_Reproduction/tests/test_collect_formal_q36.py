import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from experiments import collect_formal_q36 as collect
from experiments.formal_q36_cases import cases


class CollectorContract(unittest.TestCase):
    def test_ce_counts_inaccurate_finite_answer_as_unsuccessful(self):
        q=collect.quality(np.array([[100.,100.,100.,0.]]),np.array([[100.,109.,111.,np.nan]]),np.ones((1,4),bool))
        self.assertEqual(q['TCR'],50.)
        self.assertEqual(q['AR'],75.)
        self.assertEqual(q['CE3'],(0+9+3+3)/4)

    def test_unknown_predictions_are_false_negatives(self):
        out=collect.identification(np.array([.5,.5]),np.array([True,False]),np.zeros(2,bool),dict(low=.3,high=.8))
        self.assertEqual(out['macro_F1'],0.)
        self.assertEqual(out['unknown'],100.)

    def test_pairwise_domains_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);case=cases('primary')[0];source={'src/example.py':'1'*64}
            honest=np.arange(80)<56;certified=np.arange(80)<8
            truth=np.full((50,100),100.);mask=np.zeros((50,100),bool);mask[:,:50]=True
            jobs=[]
            for method in ['BTPWR','ETBP-TD']:
                folder=root/'cases'/case['name']/method;folder.mkdir(parents=True)
                pred=np.full((50,100),np.nan);pred[:,0]=100.;pred[:,2]=111. if method=='BTPWR' else 102.
                if method=='BTPWR':pred[:,1]=109.
                res=np.zeros((50,5))
                if method=='ETBP-TD':res[:,[0,4]]=np.nan
                reputation=np.tile(np.where(honest,.95,.1),(51,1))
                path=folder/'result.npz'
                np.savez_compressed(path,predictions=pred,truth=truth,task_mask=mask,selected=np.zeros((50,80),bool),
                                    resources=res,reputations=reputation,honest=honest,certified=certified)
                meta=dict(status='completed',rounds=50,case=case,method=method,source_sha256=source,world_sha256='shared',
                          output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),class_thresholds=dict(low=.4,high=.9,credible_strict=True))
                (folder/'result.json').write_text(json.dumps(meta),encoding='utf-8')
                jobs.append(dict(case=case['name'],method=method))
            (root/'jobs_primary.json').write_text(json.dumps(dict(jobs=jobs,source_sha256=source)),encoding='utf-8')
            summary,rounds,pairs=collect.collect(root,root/'analysis')
            self.assertEqual(len(summary),2);self.assertEqual(len(rounds),100)
            self.assertEqual(pairs.iloc[0].common_tasks,100)
            self.assertAlmostEqual(pairs.iloc[0].ours_MAE,5.5)
            self.assertAlmostEqual(pairs.iloc[0].baseline_MAE,1.)
            row=summary[summary.method=='ETBP-TD'].iloc[0]
            self.assertTrue(np.isnan(row.commitment));self.assertEqual(row.macro_F1,100.)
            path=root/'cases'/case['name']/'BTPWR'/'result.npz';path.write_bytes(path.read_bytes()+b'corrupted')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                collect.validate(path.with_name('result.json'))


if __name__=='__main__':unittest.main()
