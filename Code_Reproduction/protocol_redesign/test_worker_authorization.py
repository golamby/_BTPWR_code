import unittest
from types import SimpleNamespace
from cryptography.hazmat.primitives.asymmetric import ed25519
import securewire as w
from transport import authenticate_rpc,worker_tls_name
from services import Broker,Requester,validate_task_certificate
class WorkerAuthTests(unittest.TestCase):
 def test_tls_owner_binding_and_role_binding(self):
  key=ed25519.Ed25519PrivateKey.generate();route='ab'*32;peer=SimpleNamespace(role='broker',worker_verify_keys={route:key.public_key()},verify_keys={'controller':key.public_key()})
  message=w.sign(dict(role='worker',owner_route=route),key)
  self.assertEqual(authenticate_rpc(peer,message,worker_tls_name(route))[1],'worker:'+route)
  with self.assertRaises(ValueError):authenticate_rpc(peer,message,'controller')
  with self.assertRaises(ValueError):authenticate_rpc(peer,message,worker_tls_name('cd'*32))
  with self.assertRaises(ValueError):authenticate_rpc(peer,w.sign(dict(role='controller'),key),worker_tls_name(route))
 def test_workers_cannot_schedule_or_request_task_certificates(self):
  broker=Broker.__new__(Broker)
  for op in ['select','finish','enroll','bootstrap','metrics']:
   with self.assertRaisesRegex(ValueError,'scheduling'):broker.handle(op,{},'worker:'+'ab'*32)
  requester=Requester.__new__(Requester)
  with self.assertRaisesRegex(ValueError,'scheduling'):requester.handle('task_certificate',{'epoch':1},'worker:'+'ab'*32)
 def test_foreign_route_and_report_alias_rejected(self):
  broker=Broker.__new__(Broker);broker.workers={};broker.offers={'alias':{'route':'other'}}
  with self.assertRaisesRegex(ValueError,'Foreign worker route'):broker.handle('claim',{'signed':{'payload':{'route':'other'}}},'worker:owner')
  with self.assertRaisesRegex(ValueError,'Foreign worker report alias'):broker.handle('report',{'alias':'alias'},'worker:owner')
 def test_task_signature_epoch_budget_and_domain(self):
  key=ed25519.Ed25519PrivateKey.generate();cfg=dict(task='task',budget=450,policy={'a':1},D=100)
  context=dict(task='task',epoch=1,virtual_tasks=list(range(50)),budget=450,policy_digest=w.digest(cfg['policy']))
  cert=w.sign(context,key);validate_task_certificate(cert,key.public_key(),cfg,1)
  for altered in [dict(context,epoch=2),dict(context,budget=500),dict(context,virtual_tasks=[101]*50),dict(context,task='foreign')]:
   with self.assertRaises(ValueError):validate_task_certificate(w.sign(altered,key),key.public_key(),cfg,1)
  altered=dict(cert,payload=dict(context,virtual_tasks=[1]*50))
  with self.assertRaises(Exception):validate_task_certificate(altered,key.public_key(),cfg,1)
if __name__=='__main__':unittest.main()
