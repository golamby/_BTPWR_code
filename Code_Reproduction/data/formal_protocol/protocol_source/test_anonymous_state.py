import unittest
from cryptography.hazmat.primitives.asymmetric import ed25519
from phe import paillier
from anonymous_state import encode_score,decode_score,rerandomize,make_batch,StateRegistry,grade_code,token_header,assemble_batch,verify_state,validate_header,certificate_root
from securewire import sign,canonical
import copy

def signed_states(tokens,signer):
    return assemble_batch(tokens,[sign(token_header(t),signer) for t in tokens],signer.public_key())

class AnonymousStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test key only; deployed benchmark must use 2048-bit or stronger keys.
        cls.pub,cls.priv=paillier.generate_paillier_keypair(n_length=1024)
    def test_rerandomization_changes_ciphertext_preserves_exact_score(self):
        a=encode_score(self.pub,8734);b=rerandomize(self.pub,a)
        self.assertNotEqual(a,b);self.assertEqual(decode_score(self.priv,b),8734)
    def test_all_states_rotate_and_old_negative_state_cannot_be_reused(self):
        signer=ed25519.Ed25519PrivateKey.generate();registry=StateRegistry('task',signer.public_key())
        tokens,routes=make_batch(self.pub,{'worker':dict(ciphertext=encode_score(self.pub,9000),grade=9500)},'task',1)
        registry.install(1,signed_states(tokens,signer),routes)
        old=registry.current['worker'];registry.validate_offer('worker',old,1)
        with self.assertRaises(ValueError):registry.validate_offer('worker',old,1)
        tokens,routes=make_batch(self.pub,{'worker':dict(ciphertext=encode_score(self.pub,1000),grade=1500)},'task',2)
        registry.install(2,signed_states(tokens,signer),routes)
        with self.assertRaises(ValueError):registry.validate_offer('worker',old,2)
        registry.validate_offer('worker',registry.current['worker'],2)
    def test_grades_preserve_threshold_classes(self):
        for code in (0,2999,3000,7999,8000,10000):
            original=0 if code<3000 else 2 if code>=8000 else 1
            grade=grade_code(code)
            classified=0 if grade<3000 else 2 if grade>=8000 else 1
            self.assertEqual(original,classified)
    def test_attestation_omits_ciphertext_and_rejects_injected_fields(self):
        tokens,_=make_batch(self.pub,{'worker':dict(ciphertext=encode_score(self.pub,8734),grade=8500)},'task',1)
        header=token_header(tokens[0])
        self.assertNotIn('encrypted_reputation',header)
        self.assertNotIn(tokens[0]['encrypted_reputation'],canonical(header).decode())
        with self.assertRaises(ValueError):validate_header(dict(header,encrypted_reputation=tokens[0]['encrypted_reputation']))
    def test_certificate_binds_ciphertext_and_grade(self):
        signer=ed25519.Ed25519PrivateKey.generate()
        tokens,_=make_batch(self.pub,{'worker':dict(ciphertext=encode_score(self.pub,8734),grade=8500)},'task',1)
        original=signed_states(tokens,signer)[0]
        verify_state(original,signer.public_key())
        for field,value in [('grade',9500),('encrypted_reputation',encode_score(self.pub,1000))]:
            altered=copy.deepcopy(original);altered['payload'][field]=value
            with self.assertRaises(ValueError):verify_state(altered,signer.public_key())
    def test_public_root_uses_attestations_without_ciphertexts(self):
        signer=ed25519.Ed25519PrivateKey.generate()
        tokens,_=make_batch(self.pub,{str(i):dict(ciphertext=encode_score(self.pub,5000),grade=5500) for i in range(2)},'task',1)
        states=signed_states(tokens,signer);certs=[t['certificate'] for t in states]
        self.assertEqual(certificate_root(certs),certificate_root(certs[::-1]))
        self.assertNotIn('encrypted_reputation',canonical(certs).decode())
    def test_missing_mandatory_update_rejected(self):
        signer=ed25519.Ed25519PrivateKey.generate();registry=StateRegistry('task',signer.public_key())
        tokens,routes=make_batch(self.pub,{str(i):dict(ciphertext=encode_score(self.pub,5000),grade=5500) for i in range(2)},'task',1)
        registry.install(1,signed_states(tokens,signer),routes)
        tokens,routes=make_batch(self.pub,{'0':dict(ciphertext=encode_score(self.pub,5000),grade=5500)},'task',2)
        with self.assertRaises(ValueError):registry.install(2,signed_states(tokens,signer),routes)

if __name__=='__main__':unittest.main()
