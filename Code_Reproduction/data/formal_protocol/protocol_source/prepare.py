"""Build role-separated test credentials and a configurable client mobility replay."""
import argparse,datetime,json,secrets,sys,hashlib
from pathlib import Path
import numpy as np
from cryptography import x509
from cryptography.x509.oid import NameOID,ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import ec,ed25519,x25519
from phe import paillier
import securewire as w
from transport import save_private,worker_tls_name

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--dataset',default='tdrive');ap.add_argument('--seed',type=int,default=9801)
    ap.add_argument('--workers',type=int,choices=[50,60,70,80,90,100],default=80)
    ap.add_argument('--budget',type=float,default=450.)
    args=ap.parse_args();sys.path.insert(0,args.source)
    if args.budget<=0:raise ValueError("Positive per-round budget required")
    from src import data
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    ca_key=ec.generate_private_key(ec.SECP256R1());now=datetime.datetime.utcnow()
    name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'BTPWR isolated test CA')])
    ca=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(days=1))
        .not_valid_after(now+datetime.timedelta(days=30)).add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True)
        .sign(ca_key,hashes.SHA256()))
    (root/'ca.pem').write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    roles={}
    for role in ('authority','controller','broker','requester','ledger'):
        folder=root/role;folder.mkdir(exist_ok=True)
        signer=ed25519.Ed25519PrivateKey.generate();enc=x25519.X25519PrivateKey.generate()
        save_private(signer,folder/'sign.pem');save_private(enc,folder/'encrypt.pem')
        tls_key=ec.generate_private_key(ec.SECP256R1());save_private(tls_key,folder/'tls-key.pem')
        cert=(x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,role)]))
            .issuer_name(ca.subject).public_key(tls_key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now-datetime.timedelta(days=1)).not_valid_after(now+datetime.timedelta(days=30))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost')]),critical=False)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH,ExtendedKeyUsageOID.CLIENT_AUTH]),critical=False)
            .sign(ca_key,hashes.SHA256()))
        (folder/'tls.pem').write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        roles[role]=dict(sign=w.b64(w.public_bytes(signer.public_key())),encrypt=w.b64(w.public_bytes(enc.public_key())))
    pub,priv=paillier.generate_paillier_keypair(n_length=2048)
    (root/'requester'/'paillier-private.json').write_text(json.dumps({'p':str(priv.p),'q':str(priv.q)}))
    task=secrets.token_hex(16)
    config=dict(task=task,roles=roles,paillier_n=str(pub.n),ports=dict(broker=22401,requester=22402,ledger=22403),
                n=args.workers,D=100,T=50,budget=args.budget,eps1=2.1,eps2=2.7,grade_precision=.1,
                anonymity='fresh task-round aliases; broker routing hidden from requester; value/trajectory fingerprints allowed leakage',
                trust='honest-but-curious noncolluding broker and requester; malicious workers; trusted enrollment')
    config['client_authentication']='per-worker-mtls-owner-bound; scheduler-separated; requester-signed-task-context-v1'
    config['state_attestation']='digest-only-v1; no full ciphertexts in issuer signing requests'
    config['policy']=dict(opportunity_coefficient=3.,progress_coefficients=[.5,.5],grade_coefficients=[.5,.5],
        unknown_target=[.1,.5],credible_reference=.1,target_scale=.7,secondary_weight=.5,evidence_cap=4.,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')
            if not p.name.startswith(('draw_','plot_','reconstruct_','generate_','test_'))})
    (root/'public.json').write_text(json.dumps(config,indent=2))
    inputs=data.load_inputs()
    case=dict(name='e2e_'+args.dataset+'_'+str(args.seed)+'_n'+str(args.workers)+'_b'+format(args.budget,'g'),kind='main',stage='confirmation',group='confirmation',
              dataset=args.dataset,field='legacy',seed=args.seed,n=args.workers,H=.7,C=.1,gamma=.6,eps1=2.1,eps2=2.7,
              budget_override=args.budget,budget_factor=args.budget/data.BUDGETS[args.dataset],cohort_rule='training-top200-uniform100')
    world=data.build_world(case,inputs)
    # Only the client/auditor controls the private fixture. Broker never loads it.
    np.savez_compressed(root/'controller'/'world.npz',**{k:v for k,v in world.items() if isinstance(v,np.ndarray) or np.isscalar(v)})
    (root/'controller'/'case.json').write_text(json.dumps(case,indent=2))
    (root/'reference.json').write_text(json.dumps(dict(public=world['public_reference'].tolist(),offset=world['native_offset'],scale=world['native_scale'])))
    (root/'requester'/'task_schedule.json').write_text(json.dumps([dict(task=task,epoch=t+1,virtual_tasks=world['virtual_task_cells'][t].tolist(),budget=config['budget'],policy_digest=w.digest(config['policy'])) for t in range(config['T'])]))
    authority_key=serialization.load_pem_private_key((root/'authority'/'sign.pem').read_bytes(),password=None)
    workers=[];registry=[]
    from anonymous_state import encode_score,grade_code
    for i in range(args.workers):
        owner=ed25519.Ed25519PrivateKey.generate();enc=x25519.X25519PrivateKey.generate()
        owner_public=w.b64(w.public_bytes(owner.public_key()));route=w.digest({'owner':owner_public})
        folder=root/'controller'/'workers'/str(i);save_private(owner,folder/'sign.pem');save_private(enc,folder/'encrypt.pem')
        worker_tls=ec.generate_private_key(ec.SECP256R1());save_private(worker_tls,folder/'tls-key.pem')
        worker_cert=(x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,worker_tls_name(route))]))
            .issuer_name(ca.subject).public_key(worker_tls.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now-datetime.timedelta(days=1)).not_valid_after(now+datetime.timedelta(days=30))
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),critical=False).sign(ca_key,hashes.SHA256()))
        (folder/'tls.pem').write_bytes(worker_cert.public_bytes(serialization.Encoding.PEM))
        cert=w.sign(dict(task=task,owner_public=owner_public,owner_encrypt=w.b64(w.public_bytes(enc.public_key())),
                         initial_reputation=encode_score(pub,int(round(world['initial_reputation'][i]*10000))),
                         initial_grade=grade_code(int(round(world['initial_reputation'][i]*10000)))),authority_key)
        workers.append(dict(index=i,route=route,certificate=cert))
        registry.append(dict(synthetic_real_identity='enrollment-'+str(i),route=route))
    (root/'controller'/'workers.json').write_text(json.dumps(workers))
    (root/'authority'/'identity-registry.json').write_text(json.dumps(registry))
    print(json.dumps(dict(prepared=True,clients=args.workers,rounds=50,paillier_bits=pub.n.bit_length(),task=task)))

if __name__=='__main__':main()
