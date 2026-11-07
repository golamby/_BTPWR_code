"""Authenticated, encrypted round messages for the distributed prototype.

This transport component does not establish anonymous credential ownership.
Identity and reputation disclosure depend on the separately selected protocol.
"""
import base64
import hashlib
import json
import os
import struct
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MAX_FRAME=32*1024*1024

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def b64(value):return base64.b64encode(value).decode('ascii')
def unb64(value):return base64.b64decode(value,validate=True)
def public_bytes(key):
    return key.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)

def seal(payload,recipient_public,context):
    ephemeral=x25519.X25519PrivateKey.generate()
    shared=ephemeral.exchange(x25519.X25519PublicKey.from_public_bytes(recipient_public))
    aad=canonical(context)
    key=HKDF(algorithm=hashes.SHA256(),length=32,salt=None,info=b'BTPWR-E2E-v1'+hashlib.sha256(aad).digest()).derive(shared)
    nonce=os.urandom(12)
    return dict(ephemeral=b64(public_bytes(ephemeral.public_key())),nonce=b64(nonce),
                ciphertext=b64(AESGCM(key).encrypt(nonce,canonical(payload),aad)),context=context)

def unseal(envelope,recipient_private,context):
    if envelope['context']!=context:raise ValueError('Wrong encryption context')
    aad=canonical(context)
    shared=recipient_private.exchange(x25519.X25519PublicKey.from_public_bytes(unb64(envelope['ephemeral'])))
    key=HKDF(algorithm=hashes.SHA256(),length=32,salt=None,info=b'BTPWR-E2E-v1'+hashlib.sha256(aad).digest()).derive(shared)
    return json.loads(AESGCM(key).decrypt(unb64(envelope['nonce']),unb64(envelope['ciphertext']),aad))

def sign(payload,private_key):
    return dict(payload=payload,signature=b64(private_key.sign(canonical(payload))))

def verify(envelope,public_key):
    public_key.verify(unb64(envelope['signature']),canonical(envelope['payload']))
    return envelope['payload']

def _receive_exact(sock,n):
    parts=[]
    while n:
        part=sock.recv(n)
        if not part:raise EOFError('Connection closed inside message')
        parts.append(part);n-=len(part)
    return b''.join(parts)

def send(sock,message):
    data=canonical(message)
    if len(data)>MAX_FRAME:raise ValueError('Oversized message')
    wire=struct.pack('!I',len(data))+data;sock.sendall(wire)
    return len(wire)

def receive(sock):
    header=_receive_exact(sock,4);size=struct.unpack('!I',header)[0]
    if size>MAX_FRAME:raise ValueError('Oversized message')
    return json.loads(_receive_exact(sock,size)),size+4

def exchange(sock,message):
    started=time.perf_counter();sent=send(sock,message);reply,received=receive(sock)
    return reply,dict(seconds=time.perf_counter()-started,serialized_bytes_sent=sent,
                      serialized_bytes_received=received)
