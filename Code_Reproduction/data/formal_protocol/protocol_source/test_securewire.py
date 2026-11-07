import copy
import socket
import threading
import unittest
from cryptography.exceptions import InvalidSignature,InvalidTag
from cryptography.hazmat.primitives.asymmetric import ed25519,x25519
import securewire as w

class SecureWireTests(unittest.TestCase):
    def test_envelope_rejects_wrong_recipient_context_and_changed_ciphertext(self):
        key=x25519.X25519PrivateKey.generate();ctx={'round':1,'task':'a'}
        enc=w.seal({'report':25,'reputation':.7},w.public_bytes(key.public_key()),ctx)
        self.assertEqual(w.unseal(enc,key,ctx)['report'],25)
        with self.assertRaises(ValueError):w.unseal(enc,key,{'round':2,'task':'a'})
        with self.assertRaises(InvalidTag):w.unseal(enc,x25519.X25519PrivateKey.generate(),ctx)
        changed=copy.deepcopy(enc);cipher=bytearray(w.unb64(changed['ciphertext']));cipher[0]^=1;changed['ciphertext']=w.b64(cipher)
        with self.assertRaises(InvalidTag):w.unseal(changed,key,ctx)
    def test_signature_binds_payload(self):
        key=ed25519.Ed25519PrivateKey.generate();m=w.sign({'round':1,'root':'abc'},key)
        self.assertEqual(w.verify(m,key.public_key())['round'],1)
        m['payload']['root']='def'
        with self.assertRaises(InvalidSignature):w.verify(m,key.public_key())
    def test_actual_socket_frame_and_size_receipt(self):
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1)
        def server():
            conn,_=listener.accept()
            with conn:
                msg,count=w.receive(conn);w.send(conn,{'received':count,'nonce':msg['nonce']})
        worker=threading.Thread(target=server);worker.start()
        with socket.create_connection(listener.getsockname()) as conn:
            reply,receipt=w.exchange(conn,{'nonce':'fixed-test'})
        worker.join();listener.close()
        self.assertEqual(reply['received'],receipt['serialized_bytes_sent'])
        self.assertEqual(reply['nonce'],'fixed-test')

if __name__=='__main__':unittest.main()
