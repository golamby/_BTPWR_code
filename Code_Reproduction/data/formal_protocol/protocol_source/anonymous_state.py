"""Paillier re-randomized, issuer-signed reputation tokens.

The honest-but-curious broker knows routing. Requesters see scores but not
the routing table. Score/trajectory fingerprint inference is outside this
cryptographic unlinkability property.
"""
import secrets
from phe import paillier
from securewire import sign,verify,digest

HEADER_FIELDS={'kind','task','epoch','serial','grade','token_digest'}
TOKEN_FIELDS={'task','epoch','serial','grade','encrypted_reputation'}

def validate_header(header):
    if set(header)!=HEADER_FIELDS or header['kind']!='reputation-state-v1':raise ValueError('Invalid attestation schema')
    if not isinstance(header['task'],str) or not header['task']:raise ValueError('Invalid task')
    if type(header['epoch']) is not int or header['epoch']<1:raise ValueError('Invalid epoch')
    if type(header['grade']) is not int or header['grade'] not in range(500,10000,1000):raise ValueError('Invalid grade')
    for field in ('serial','token_digest'):
        value=header[field]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid '+field)
    return header

def token_header(token):
    """Only this digest-bearing header is sent to the decryption-key holder."""
    if set(token)!=TOKEN_FIELDS or not isinstance(token['encrypted_reputation'],str):raise ValueError('Invalid token schema')
    return validate_header(dict(kind='reputation-state-v1',task=token['task'],epoch=token['epoch'],
        serial=token['serial'],grade=token['grade'],token_digest=digest(token)))

def verify_state(envelope,issuer_public):
    if set(envelope)!={'payload','certificate'}:raise ValueError('Invalid state envelope')
    token=envelope['payload']
    if verify(envelope['certificate'],issuer_public)!=token_header(token):raise ValueError('State digest/context mismatch')
    return token

def assemble_batch(tokens,certificates,issuer_public):
    by_serial={c['payload']['serial']:c for c in certificates}
    if len(by_serial)!=len(tokens) or len(certificates)!=len(tokens):raise ValueError('Certificate batch mismatch')
    result=[]
    for token in tokens:
        envelope=dict(payload=token,certificate=by_serial[token['serial']])
        verify_state(envelope,issuer_public);result.append(envelope)
    return result

def certificate_root(certificates):
    return digest(sorted(certificates,key=lambda c:c['payload']['serial']))

def grade_code(code):
    """Public 0.1-wide midpoint grade, preserving the 0.3/0.8 classes."""
    if type(code) is not int or not 0<=code<=10000:raise ValueError('Invalid score code')
    return min(9500,(code//1000)*1000+500)

def encode_score(public_key,code):
    if type(code) is not int or not 0<=code<=10000:raise ValueError('Reputation code out of range')
    return format(public_key.encrypt(code).ciphertext(),'0{}x'.format((public_key.n.bit_length()+7)//8*4))

def decode_score(private_key,ciphertext):
    value=int(ciphertext,16);pub=private_key.public_key
    if not 0<value<pub.nsquare:raise ValueError('Invalid ciphertext')
    code=private_key.decrypt(paillier.EncryptedNumber(pub,value,0))
    if type(code) is not int or not 0<=code<=10000:raise ValueError('Invalid reputation plaintext')
    return code

def rerandomize(public_key,ciphertext):
    value=int(ciphertext,16)
    if not 0<value<public_key.nsquare:raise ValueError('Invalid ciphertext')
    encrypted=paillier.EncryptedNumber(public_key,value,0)
    encrypted.obfuscate()
    return format(encrypted.ciphertext(),'0{}x'.format((public_key.n.bit_length()+7)//8*4))

def make_batch(public_key,records,task,epoch):
    """Return shuffled fresh tokens and a broker-only routing table."""
    entries=[]
    for route,record in records.items():
        ciphertext=record['ciphertext'];grade=record['grade']
        if grade not in range(500,10000,1000):raise ValueError('Invalid public grade')
        token=dict(task=task,epoch=epoch,serial=secrets.token_hex(32),
                   encrypted_reputation=rerandomize(public_key,ciphertext),grade=grade)
        entries.append((route,token))
    secrets.SystemRandom().shuffle(entries)
    return [t for _,t in entries],{t['serial']:route for route,t in entries}

class StateRegistry:
    def __init__(self,task,issuer_public):
        self.task=task;self.issuer_public=issuer_public;self.epoch=0
        self.current={};self.used_offers=set();self.spent=set()
    def install(self,epoch,signed_tokens,routes):
        if epoch!=self.epoch+1:raise ValueError('Nonsequential epoch')
        next_tokens={}
        for envelope in signed_tokens:
            token=verify_state(envelope,self.issuer_public)
            if token['task']!=self.task or token['epoch']!=epoch:raise ValueError('Wrong token context')
            if token['grade'] not in range(500,10000,1000):raise ValueError('Invalid grade')
            if token['serial'] not in routes or token['serial'] in self.spent:raise ValueError('Unknown/reused token')
            route=routes[token['serial']]
            if route in next_tokens:raise ValueError('Duplicate owner state')
            next_tokens[route]=envelope
        if self.current and set(next_tokens)!=set(self.current):raise ValueError('Missing mandatory successor')
        for old in self.current.values():self.spent.add(old['payload']['serial'])
        self.current=next_tokens;self.epoch=epoch;self.used_offers=set()
    def validate_offer(self,route,token,round_number):
        if round_number!=self.epoch:raise ValueError('Wrong round')
        if route not in self.current or token!=self.current[route]:raise ValueError('Stale/foreign state')
        if token['payload']['serial'] in self.spent:raise ValueError('Consumed state')
        if route in self.used_offers:raise ValueError('Duplicate offer')
        self.used_offers.add(route)
        return token['payload']['encrypted_reputation']
