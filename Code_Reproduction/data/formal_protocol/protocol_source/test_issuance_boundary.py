import unittest
from types import SimpleNamespace
from cryptography.hazmat.primitives.asymmetric import ed25519
from anonymous_state import token_header,encode_score
from phe import paillier
from services import Broker,Requester

class IssuanceBoundaryTests(unittest.TestCase):
    def test_open_rejects_ciphertext_before_lock(self):
        requester=Requester.__new__(Requester);requester.round=None
        requester.peer=SimpleNamespace(call=lambda *args:dict(epoch=0))
        with self.assertRaisesRegex(ValueError,'confidential state'):
            requester.handle('open_round',dict(epoch=1,items=[dict(alias='worker',ciphertext='hidden')]),'broker')

    def test_signing_api_rejects_full_tokens(self):
        requester=Requester.__new__(Requester);requester.round=None
        requester.peer=SimpleNamespace(config=dict(n=1,task='task'),signer=ed25519.Ed25519PrivateKey.generate())
        token=dict(task='task',epoch=1,serial='aa'*32,grade=5500,encrypted_reputation='not-for-issuer')
        with self.assertRaises(ValueError):requester.handle('sign_batch',dict(tokens=[token],bootstrap=True),'broker')
        header=dict(token_header(token),encrypted_reputation=token['encrypted_reputation'])
        with self.assertRaises(ValueError):requester.handle('sign_batch',dict(headers=[header],bootstrap=True),'broker')

    def test_missing_negative_update_cannot_silently_carry_old_state(self):
        broker=Broker.__new__(Broker)
        broker.registry=SimpleNamespace(epoch=1);broker.selection=['selected']
        broker.reports={str(i):{} for i in range(99)};broker.reports['selected']={}
        broker.pub,_=paillier.generate_paillier_keypair(n_length=1024)
        broker.offers={'selected':dict(ciphertext=encode_score(broker.pub,5000))}
        broker.peer=SimpleNamespace(config=dict(n=100),call=lambda *args:dict(ciphertexts={},grades={}))
        with self.assertRaisesRegex(ValueError,'mandatory selected update'):
            broker.handle('finish',{},'controller')

if __name__=='__main__':unittest.main()
