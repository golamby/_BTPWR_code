"""Role-separated requester and broker services for the authorized-reader model."""
import argparse,json,secrets,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from cryptography.hazmat.primitives.asymmetric import ed25519
from phe import paillier
import securewire as w
from transport import Peer,serve
from anonymous_state import (StateRegistry,encode_score,decode_score,rerandomize,make_batch,grade_code,
                             token_header,validate_header,assemble_batch,certificate_root)

def state_root(tokens):return certificate_root([t['certificate'] for t in tokens])
def report_unpack(encoded,D):
    data=w.unb64(encoded)
    if len(data)!=D*5:raise ValueError('Wrong fixed report length')
    mask=np.frombuffer(data[:D],dtype=np.uint8).astype(bool)
    value=np.frombuffer(data[D:],dtype='<f4').astype(float)
    if not np.isfinite(value).all() or np.any((value<10)|(value>40)):raise ValueError('Report range')
    return np.where(mask,value,np.nan)

def validate_task_certificate(certificate,public_key,config,epoch):
    context=w.verify(certificate,public_key)
    if set(context)!={'task','epoch','virtual_tasks','budget','policy_digest'}:raise ValueError('Task context schema')
    if context['task']!=config['task'] or context['epoch']!=epoch:raise ValueError('Task context epoch')
    if context['budget']!=config['budget'] or context['policy_digest']!=w.digest(config['policy']):raise ValueError('Task budget/policy binding')
    tasks=context['virtual_tasks']
    if not isinstance(tasks,list) or len(tasks)!=50 or any(type(x) is not int or not 0<=x<config['D'] for x in tasks):raise ValueError('Task token domain')
    return context

class Requester:
    def __init__(self,peer):
        from src import recovery
        self.peer=peer;self.pub=paillier.PaillierPublicKey(int(peer.config['paillier_n']))
        secret=json.loads((peer.root/'requester'/'paillier-private.json').read_text())
        self.priv=paillier.PaillierPrivateKey(self.pub,int(secret['p']),int(secret['q']))
        public=json.loads((peer.root/'reference.json').read_text());self.public=public
        self.head=recovery.initialize(np.asarray(public['public']),public['offset'],public['scale'])
        self.round=None;self.pending=None;self.values={}
        self.task_schedule=json.loads((peer.root/'requester'/'task_schedule.json').read_text())
    def handle(self,op,b,role):
        if op=='task_certificate':
            if role!='controller':raise ValueError('Task scheduling authorization')
            if set(b)!={'epoch'} or not 1<=b['epoch']<=len(self.task_schedule):raise ValueError('Task scheduling epoch')
            return dict(certificate=w.sign(self.task_schedule[b['epoch']-1],self.peer.signer))
        if role!='broker':raise ValueError('Requester accepts broker messages only')
        if op=='open_round':
            if self.round is not None:raise ValueError('Round active')
            ledger=self.peer.call('ledger','state',{})
            if b['epoch']!=ledger['epoch']+1:raise ValueError('Epoch mismatch')
            if any(set(item)!={'alias'} for item in b['items']):raise ValueError('Open contains confidential state')
            context=validate_task_certificate(b['task_certificate'],self.peer.verify_keys['requester'],self.peer.config,b['epoch'])
            if context!=self.task_schedule[b['epoch']-1]:raise ValueError('Unregistered task schedule')
            expected=w.digest(dict(offers_digest=b['offers_digest'],task_certificate_digest=w.digest(b['task_certificate'])))
            if expected!=b['offer_root']:raise ValueError('Task/offer root binding')
            self.round=b
            if len({x['alias'] for x in b['items']})!=len(b['items']):raise ValueError('Duplicate selected alias')
            self.peer.call('ledger','open',dict(epoch=b['epoch'],offers=b['offer_root'],previous=ledger['root'],budget=int(round(self.peer.config['budget']*1000000))))
            return dict(opened=True)
        if op=='evaluate':
            from src import trust,recovery
            from src.count_feedback import feedback
            if self.round is None or b['epoch']!=self.round['epoch']:raise ValueError('No matching round')
            ledger=self.peer.call('ledger','state',{})
            if ledger['phase']!=2 or ledger['epoch']!=b['epoch']:raise ValueError('Roster not locked')
            aliases=[x['alias'] for x in self.round['items']];index={a:i for i,a in enumerate(aliases)}
            if set(aliases)!=set(b['selected']):raise ValueError('Changed selected cohort')
            if len(b['states'])!=len(aliases) or {x['alias'] for x in b['states']}!=set(aliases):raise ValueError('Selected score cohort mismatch')
            self.values={x['alias']:decode_score(self.priv,x['ciphertext'])/10000 for x in b['states']}
            r=np.array([self.values[a] for a in aliases]);raw=np.full((len(aliases),self.peer.config['D']),np.nan)
            received=set()
            for record in b['reports']:
                alias=record['alias']
                if alias not in index or alias not in b['selected']:raise ValueError('Unselected report')
                if alias in received:raise ValueError('Duplicate report')
                if w.digest({'key':record['alias_public']})!=alias:raise ValueError('Report key binding')
                received.add(alias)
                public=ed25519.Ed25519PublicKey.from_public_bytes(w.unb64(record['alias_public']))
                body=w.verify(record['signed'],public)
                if body['epoch']!=b['epoch'] or body['alias']!=alias:raise ValueError('Report scope')
                context=dict(task=self.peer.config['task'],epoch=b['epoch'],alias=alias,kind='report')
                payload=w.unseal(body['sealed'],self.peer.decryptor,context)
                raw[index[alias]]=report_unpack(payload['fixed'],self.peer.config['D'])
            if received!=set(aliases):raise ValueError('Missing selected report')
            from src.signed_grade_policy import requester_round
            started=time.perf_counter();updated=requester_round(raw,r,self.head)
            service=updated['prediction']
            seconds=time.perf_counter()-started
            cipher_by_alias={a:encode_score(self.pub,int(round(updated['next_reputation'][i]*10000))) for i,a in enumerate(aliases)}
            scores=w.digest(cipher_by_alias);reports_root=w.digest(b['reports'])
            self.pending=dict(epoch=b['epoch'],reports=reports_root,scores=scores)
            # Only the authorized test auditor receives service values. Broker sees an envelope.
            audit=dict(predictions=[None if not np.isfinite(v) else float(v) for v in service],
                       reputations={a:float(updated['next_reputation'][i]) for i,a in enumerate(aliases)},compute_seconds=seconds)
            enc=w.seal(audit,w.unb64(self.peer.config['roles']['controller']['encrypt']),
                       dict(task=self.peer.config['task'],epoch=b['epoch'],kind='audit'))
            return dict(ciphertexts=cipher_by_alias,
                        grades={a:grade_code(int(round(updated['next_reputation'][i]*10000))) for i,a in enumerate(aliases)},
                        report_root=reports_root,scores_root=scores,audit=enc)
        if op=='sign_batch':
            if set(b)-{'headers','bootstrap'}:raise ValueError('Signing request contains non-header fields')
            if len(b['headers'])!=self.peer.config['n']:raise ValueError('Wrong batch size')
            headers=[validate_header(t) for t in b['headers']]
            epoch=headers[0]['epoch'];bootstrap=b.get('bootstrap',False)
            if len({t['serial'] for t in headers})!=len(headers):raise ValueError('Duplicate token serial')
            if bootstrap:
                if epoch!=1 or self.round is not None:raise ValueError('Bad bootstrap')
            elif self.pending is None or epoch!=self.pending['epoch']+1:raise ValueError('Missing scoring phase')
            for token in headers:
                if token['task']!=self.peer.config['task'] or token['epoch']!=epoch:raise ValueError('Token scope')
            certificates=[w.sign(t,self.peer.signer) for t in headers]
            root=certificate_root(certificates)
            if not bootstrap:
                state=self.peer.call('ledger','state',{})
                self.peer.call('ledger','settle',dict(epoch=self.pending['epoch'],previous=state['root'],roster=state['roster'],
                    reports=self.pending['reports'],scores=self.pending['scores'],next=root))
                self.round=None;self.pending=None;self.values={}
            return dict(certificates=certificates,root=root)
        if op=='metrics':return self.peer.metrics
        raise ValueError('Unknown operation')

class Broker:
    def __init__(self,peer):
        self.peer=peer;self.pub=paillier.PaillierPublicKey(int(peer.config['paillier_n']))
        self.registry=StateRegistry(peer.config['task'],peer.verify_keys['requester'])
        self.workers={};self.offers={};self.reports={};self.used_aliases=set();self.selection=[]
    def handle(self,op,b,role):
        worker_route=role[7:] if role.startswith('worker:') else None
        if worker_route is not None:
            if op not in ('claim','offer','report'):raise ValueError('Worker scheduling operation forbidden')
        elif role!='controller' or op in ('claim','offer','report'):raise ValueError('Unexpected client role')
        if op=='enroll':
            cert=w.verify(b['certificate'],self.peer.verify_keys['authority'])
            if cert['task']!=self.peer.config['task']:raise ValueError('Wrong enrollment scope')
            route=w.digest({'owner':cert['owner_public']})
            if route in self.workers:raise ValueError('Repeated enrollment')
            self.workers[route]=cert
            self.peer.worker_verify_keys[route]=ed25519.Ed25519PublicKey.from_public_bytes(w.unb64(cert['owner_public']))
            return {'enrolled':True}
        if op=='bootstrap':
            if len(self.workers)!=self.peer.config['n'] or self.registry.epoch:raise ValueError('Enrollment incomplete/repeated')
            batch,routes=make_batch(self.pub,{a:dict(ciphertext=v['initial_reputation'],grade=v['initial_grade']) for a,v in self.workers.items()},self.peer.config['task'],1)
            signed=self.peer.call('requester','sign_batch',dict(headers=[token_header(t) for t in batch],bootstrap=True))
            tokens=assemble_batch(batch,signed['certificates'],self.peer.verify_keys['requester'])
            if state_root(tokens)!=signed['root']:raise ValueError('Batch root mismatch')
            self.registry.install(1,tokens,routes)
            return self.peer.call('ledger','deploy',dict(genesis=signed['root'],policy=w.digest(self.peer.config)))
        if op=='claim':
            signed=b['signed'];body=signed['payload'];route=body['route']
            if route!=worker_route:raise ValueError('Foreign worker route')
            cert=self.workers[route]
            w.verify(signed,ed25519.Ed25519PublicKey.from_public_bytes(w.unb64(cert['owner_public'])))
            if body['epoch']!=self.registry.epoch:raise ValueError('Stale claim')
            envelope=w.seal(self.registry.current[route],w.unb64(cert['owner_encrypt']),
                dict(task=self.peer.config['task'],epoch=self.registry.epoch,kind='claim'))
            return envelope
        if op=='offer':
            context=dict(task=self.peer.config['task'],epoch=self.registry.epoch,kind='offer')
            signed=w.unseal(b['sealed'],self.peer.decryptor,context);body=signed['payload'];route=body['route']
            if route!=worker_route:raise ValueError('Foreign worker route')
            cert=self.workers[route]
            w.verify(signed,ed25519.Ed25519PublicKey.from_public_bytes(w.unb64(cert['owner_public'])))
            if route!=worker_route:raise ValueError('Foreign worker route')
            cipher=self.registry.validate_offer(route,body['token'],body['epoch'])
            alias=body['alias']
            if alias in self.used_aliases or alias in self.offers:raise ValueError('Repeated alias')
            if w.digest({'key':body['alias_public']})!=alias:raise ValueError('Alias ownership')
            self.used_aliases.add(alias);self.offers[alias]=dict(body,route=route,ciphertext=cipher)
            return dict(accepted=True)
        if op=='select':
            from src import privacy,trust
            from src.fast_qualification import DynamicBalanced
            if len(self.offers)!=self.peer.config['n']:raise ValueError('Expected configured padded client cohort')
            aliases=sorted(self.offers);offers=[self.offers[a] for a in aliases]
            epoch=self.registry.epoch
            context=validate_task_certificate(b['task_certificate'],self.peer.verify_keys['requester'],self.peer.config,epoch)
            r=np.array([x['token']['payload']['grade']/10000. for x in offers])
            packet={k:np.array([v['packet'][k] for v in offers],bool) for k in ('bits','hi','nonzero')}
            eligible=np.array([v['active'] for v in offers],bool)
            from src.signed_grade_policy import select_round
            result=select_round(packet,eligible,r,context['virtual_tasks'],self.peer.config['budget'],self.peer.config['T']+1-epoch,self.peer.config['eps1'],self.peer.config['eps2'])
            ids=result['members'].tolist()
            self.selection=[aliases[i] for i in ids];self.aliases=aliases
            # Even encrypted scores are withheld from their key holder until lock.
            items=[dict(alias=a) for a in self.selection]
            offers_digest=w.digest([dict(alias=a,active=x['active'],packet=x['packet']) for a,x in zip(aliases,offers)])
            offer_root=w.digest(dict(offers_digest=offers_digest,task_certificate_digest=w.digest(b['task_certificate'])))
            self.peer.call('requester','open_round',dict(epoch=epoch,items=items,offers_digest=offers_digest,task_certificate=b['task_certificate'],offer_root=offer_root))
            amount=int(round(result['cost_numerator']/result['cost_denominator']*1000000))
            addresses=sorted(['0x'+a[-40:] for a in self.selection],key=lambda a:int(a,16))
            amounts=[amount//len(addresses)]*len(addresses) if addresses else []
            if amounts:amounts[-1]+=amount-sum(amounts)
            self.peer.call('ledger','lock',dict(roster=addresses,amounts=amounts))
            self.charge=amount/1000000
            return dict(selected=self.selection,charge=self.charge,active_packet_tie_collisions=result['active_packet_tie_collisions'])
        if op=='report':
            alias=b['alias'];offer=self.offers[alias]
            if offer['route']!=worker_route:raise ValueError('Foreign worker report alias')
            body=w.verify(b['signed'],ed25519.Ed25519PublicKey.from_public_bytes(w.unb64(offer['alias_public'])))
            if body['epoch']!=self.registry.epoch or body['alias']!=alias or alias in self.reports:raise ValueError('Report scope/replay')
            self.reports[alias]=dict(alias=alias,alias_public=offer['alias_public'],signed=b['signed'])
            return dict(accepted=True)
        if op=='finish':
            if len(self.reports)!=self.peer.config['n']:raise ValueError('Incomplete padded reporting phase')
            epoch=self.registry.epoch
            result=self.peer.call('requester','evaluate',dict(epoch=epoch,selected=self.selection,
                                  states=[dict(alias=a,ciphertext=rerandomize(self.pub,self.offers[a]['ciphertext'])) for a in self.selection],
                                  reports=[self.reports[a] for a in sorted(self.selection)]))
            if set(result['ciphertexts'])!=set(self.selection) or set(result['grades'])!=set(self.selection):
                raise ValueError('Missing or foreign mandatory selected update')
            records={}
            for a,offer in self.offers.items():
                old=offer['token']['payload']
                records[offer['route']]=dict(ciphertext=result['ciphertexts'].get(a,old['encrypted_reputation']),
                                            grade=result['grades'].get(a,old['grade']))
            batch,routes=make_batch(self.pub,records,self.peer.config['task'],epoch+1)
            signed=self.peer.call('requester','sign_batch',dict(headers=[token_header(t) for t in batch]))
            tokens=assemble_batch(batch,signed['certificates'],self.peer.verify_keys['requester'])
            if state_root(tokens)!=signed['root']:raise ValueError('Batch root mismatch')
            self.registry.install(epoch+1,tokens,routes)
            answer=dict(audit=result['audit'],root=signed['root'],epoch=epoch,selected=self.selection,charge=self.charge)
            self.offers={};self.reports={};self.selection=[]
            return answer
        if op=='metrics':
            return dict(broker=self.peer.metrics,requester=self.peer.call('requester','metrics',{}),
                        ledger=self.peer.call('ledger','receipts',{}))
        raise ValueError('Unknown operation')

def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['requester','broker','ledger']);p.add_argument('--root',required=True);p.add_argument('--source',required=True);a=p.parse_args()
    sys.path.insert(0,a.source);peer=Peer(a.root,a.role)
    if a.role=='requester':handler=Requester(peer).handle
    elif a.role=='broker':handler=Broker(peer).handle
    else:
        from evm_ledger import Ledger
        ledger=Ledger(Path(__file__).with_name('contract.json'));handler=ledger.dispatch
    serve(peer,handler)

if __name__=='__main__':main()
