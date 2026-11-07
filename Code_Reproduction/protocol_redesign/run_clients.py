"""Configurable separate client states using real TLS calls; evaluation stays at the auditor."""
import argparse,json,secrets,sys,time,resource
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from cryptography.hazmat.primitives.asymmetric import ed25519
import securewire as w
from transport import Peer,private

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--rounds',type=int,default=50);p.add_argument('--output',required=True);a=p.parse_args()
    client_setup_started=time.perf_counter()
    sys.path.insert(0,a.source)
    from src import privacy,recovery
    from src.signed_grade_policy import select_round,requester_round
    from anonymous_state import grade_code
    peer=Peer(a.root,'controller');root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    workers=json.loads((root/'controller'/'workers.json').read_text())
    world=np.load(root/'controller'/'world.npz');signers={};decryptors={}
    n=peer.config['n'];D=peer.config['D']
    if len(workers)!=n or not 1<=a.rounds<=peer.config['T']:raise ValueError('Client count/round mismatch')
    for worker in workers:
        i=worker['index'];signers[i]=private(root/'controller'/'workers'/str(i)/'sign.pem');decryptors[i]=private(root/'controller'/'workers'/str(i)/'encrypt.pem')
    worker_peers={i:Peer(a.root,'worker',worker_index=i) for i in range(n)}
    pool=ThreadPoolExecutor(max_workers=8)
    client_setup_seconds=time.perf_counter()-client_setup_started
    started=time.perf_counter()
    list(pool.map(lambda worker:peer.call('broker','enroll',dict(certificate=worker['certificate'])),workers))
    peer.call('broker','bootstrap',{})
    bootstrap_seconds=time.perf_counter()-started
    predictions=[];reputations=[];resources=[];rounds=[];client_tokens={}
    cfg=privacy.Config(n=n,m=D,T=peer.config['T'],eps1=peer.config['eps1'],eps2=peer.config['eps2'])
    current_reputation=world['initial_reputation'].copy()
    reference_r=np.rint(current_reputation*10000)/10000
    public=json.loads((root/'reference.json').read_text())
    reference_head=recovery.initialize(np.asarray(public['public']),public['offset'],public['scale'])
    checks=[];reference_seconds=0.;task_certificates=[]
    for t in range(a.rounds):
        epoch=t+1;round_start=time.perf_counter();cpu_start=time.process_time();phase={};tick=round_start;alias_signers={};aliases={}
        def claim(worker):
            i=worker['index'];payload=dict(route=worker['route'],epoch=epoch)
            enc=worker_peers[i].call('broker','claim',dict(signed=w.sign(payload,signers[i])))
            token=w.unseal(enc,decryptors[i],dict(task=peer.config['task'],epoch=epoch,kind='claim'))
            client_tokens[i]=token
        list(pool.map(claim,workers))
        phase['claim_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        loc=world['presence'][t];cost=world['actual_cost'][t];virtual=world['virtual_presence'][t]
        avg=np.divide(cost.sum(1),loc.sum(1),out=np.full(n,3.25),where=loc.sum(1)>0)
        packet=privacy.release(virtual,np.where(virtual,avg[:,None],0),world['RR_coins'][t],cfg.eps1,cfg.eps2,cfg)
        for k in ('bits','hi','nonzero'):packet[k][~world['offer_mask'][t]]=False
        def offer(worker):
            i=worker['index'];key=ed25519.Ed25519PrivateKey.generate();pub=w.b64(w.public_bytes(key.public_key()));alias=w.digest({'key':pub})
            alias_signers[i]=key;aliases[i]=alias
            body=dict(route=worker['route'],epoch=epoch,token=client_tokens[i],alias=alias,alias_public=pub,
                active=bool(world['offer_mask'][t,i]),packet={k:packet[k][i].astype(int).tolist() for k in ('bits','hi','nonzero')})
            enc=w.seal(w.sign(body,signers[i]),w.unb64(peer.config['roles']['broker']['encrypt']),
                dict(task=peer.config['task'],epoch=epoch,kind='offer'))
            worker_peers[i].call('broker','offer',dict(sealed=enc))
        list(pool.map(offer,workers))
        phase['offer_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        # Recruitment packets and task tokens must use the same virtual grid.
        # Actual region identifiers are used only inside authorized recovery/evaluation.
        task_certificate=peer.call('requester','task_certificate',dict(epoch=epoch))['certificate']
        task_certificates.append(task_certificate)
        selection=peer.call('broker','select',dict(task_certificate=task_certificate))
        selected=set(selection['selected'])
        phase['selection_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        def report(worker):
            i=worker['index'];alias=aliases[i]
            raw=world['reports_check'][t,i] if alias in selected else np.full(D,np.nan)
            mask=np.isfinite(raw).astype(np.uint8);values=np.clip(np.nan_to_num(raw,nan=25.),10.,40.).astype('<f4')
            fixed=w.b64(mask.tobytes()+values.tobytes())
            context=dict(task=peer.config['task'],epoch=epoch,alias=alias,kind='report')
            enc=w.seal(dict(fixed=fixed),w.unb64(peer.config['roles']['requester']['encrypt']),context)
            signed=w.sign(dict(epoch=epoch,alias=alias,sealed=enc),alias_signers[i])
            worker_peers[i].call('broker','report',dict(alias=alias,signed=signed))
        list(pool.map(report,workers))
        phase['report_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
        result=peer.call('broker','finish',{})
        phase['finish_seconds']=time.perf_counter()-tick
        measured_seconds=time.perf_counter()-round_start
        measured_cpu=time.process_time()-cpu_start
        audit=w.unseal(result['audit'],peer.decryptor,dict(task=peer.config['task'],epoch=epoch,kind='audit'))
        pred=np.array([np.nan if x is None else x for x in audit['predictions']]);predictions.append(pred)
        for i in range(n):
            if aliases[i] in audit['reputations']:current_reputation[i]=audit['reputations'][aliases[i]]
        reputations.append(current_reputation.copy())
        ids=[i for i in range(n) if aliases[i] in selected]
        resources.append([selection['charge'],float(cost[ids].sum()),len(ids)])
        rounds.append(dict(round=epoch,seconds=measured_seconds,controller_cpu_seconds=measured_cpu,controller_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,selected=len(ids),requester_compute_seconds=audit['compute_seconds'],**phase))
        # Offline auditor replay is outside measured protocol round time.
        audit_start=time.perf_counter()
        grades=np.asarray([grade_code(int(round(v*10000)))/10000. for v in reference_r])
        expected=select_round(packet,world['offer_mask'][t],grades,world['virtual_task_cells'][t],peer.config['budget'],peer.config['T']-t,cfg.eps1,cfg.eps2)
        exp_ids=expected['members'];assert set(exp_ids)==set(ids),'Selection mismatch'
        expected_charge=round(expected['cost_numerator']/expected['cost_denominator']*1000000)/1000000
        assert expected_charge==selection['charge'],'Charge mismatch'
        expected_raw=np.clip(world['reports_check'][t,exp_ids],10,40).astype(np.float32).astype(float)
        ref=requester_round(expected_raw,reference_r[exp_ids],reference_head)
        reference_r[exp_ids]=ref['next_reputation']
        assert np.array_equal(reference_r,current_reputation),'Reputation mismatch'
        assert np.allclose(pred,ref['prediction'],rtol=0,atol=1e-8,equal_nan=True),'Service mismatch'
        maxdiff=float(np.nanmax(np.abs(pred-ref['prediction']))) if np.any(np.isfinite(pred)) else 0.
        checks.append(dict(round=epoch,selection_equal=True,reputation_exact=True,charge_equal=True,prediction_max_abs_diff=maxdiff,active_packet_tie_collisions=expected['active_packet_tie_collisions']))
        reference_seconds+=time.perf_counter()-audit_start
        (out/'progress.json').write_text(json.dumps(dict(completed_rounds=epoch,workers=n,last_round_seconds=measured_seconds)))
        print(json.dumps(rounds[-1]),flush=True)
    pool.shutdown()
    metrics=peer.call('broker','metrics',{})
    for row in metrics.get('broker',[]):
        if row.get('source','').startswith('worker:'):row['source']='worker'
    (out/'network.json').write_text(json.dumps(dict(controller=peer.metrics,workers=[row for p in worker_peers.values() for row in p.metrics],**metrics),indent=2))
    (out/'timing.json').write_text(json.dumps(dict(client_setup_seconds=client_setup_seconds,bootstrap_seconds=bootstrap_seconds,rounds=rounds,
        total_seconds=time.perf_counter()-started,topology='separate services over mutually authenticated TLS/TCP on one host',
        communication_units='serialized authenticated/encrypted application frames; TLS record overhead excluded',
        workers=n,completed_rounds=a.rounds,offline_reference_seconds=reference_seconds,protocol_round_sum_seconds=sum(x['seconds'] for x in rounds)),indent=2))
    (out/'reference_checks.json').write_text(json.dumps(checks,indent=2))
    (out/'task_certificates.json').write_text(json.dumps(task_certificates,indent=2))
    np.savez_compressed(out/'results.npz',predictions=np.asarray(predictions),reputations=np.asarray(reputations),
        resources=np.asarray(resources),truth=world['truth_native_private'][:,:a.rounds].T,task_mask=world['task_mask'][:a.rounds],
        honest=world['honest_private'])
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
