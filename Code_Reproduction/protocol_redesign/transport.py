"""Mutually authenticated TLS RPC with signed payloads and measured frame sizes."""
import json,socket,socketserver,ssl,threading,time,secrets
try:
    import resource
except ImportError:
    resource=None
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519,x25519
import securewire as w

def private(path):return serialization.load_pem_private_key(Path(path).read_bytes(),password=None)
def save_private(key,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    p.chmod(0o600)

class Peer:
    def __init__(self,root,role,worker_index=None):
        self.root=Path(root);self.role=role
        self.config=json.loads((self.root/'public.json').read_text())
        folder=self.root/role if role!='worker' else self.root/'controller'/'workers'/str(worker_index)
        self.signer=private(folder/'sign.pem')
        self.decryptor=private(folder/'encrypt.pem')
        self.worker_verify_keys={}
        self.owner_route=w.digest({'owner':w.b64(w.public_bytes(self.signer.public_key()))}) if role=='worker' else None
        self.verify_keys={k:ed25519.Ed25519PublicKey.from_public_bytes(w.unb64(v['sign'])) for k,v in self.config['roles'].items()}
        self.context=ssl.create_default_context(cafile=str(self.root/'ca.pem'))
        self.context.load_cert_chain(str(folder/'tls.pem'),str(folder/'tls-key.pem'))
        self.context.minimum_version=ssl.TLSVersion.TLSv1_2
        self.metrics=[];self.seen=set();self.lock=threading.RLock()
    def call(self,role,operation,body,port=None):
        port=port or self.config.get('proxy_ports',self.config['ports'])[role]
        request=dict(role=self.role,operation=operation,body=body,request_id=secrets.token_hex(16),task=self.config['task'])
        if self.role=='worker':request['owner_route']=self.owner_route
        start=time.perf_counter()
        with socket.create_connection(('127.0.0.1',port),timeout=300) as plain:
            tcp=time.perf_counter()
            with self.context.wrap_socket(plain,server_hostname='localhost') as sock:
                tls=time.perf_counter();reply,m=w.exchange(sock,w.sign(request,self.signer))
        payload=w.verify(reply,self.verify_keys[role])
        if payload['request_id']!=request['request_id']:raise ValueError('Mismatched response')
        self.metrics.append(dict(target=role,operation=operation,tcp_seconds=tcp-start,
            tls_seconds=tls-tcp,total_seconds=time.perf_counter()-start,**m))
        if not payload['ok']:raise ValueError(payload['error'])
        return payload['result']
    def export_metrics(self,path):Path(path).write_text(json.dumps(self.metrics,indent=2))

def worker_tls_name(route):
    return 'worker-'+w.b64(bytes.fromhex(route)).replace('+','-').replace('/','_').rstrip('=')

def authenticate_rpc(peer,message,tls_common_name):
    payload=message['payload'];claimed=payload['role']
    if claimed=='worker':
        route=payload.get('owner_route')
        if peer.role!='broker' or tls_common_name!=worker_tls_name(route):raise ValueError('Worker TLS owner binding')
        key=peer.worker_verify_keys.get(route)
        if key is None:raise ValueError('Unenrolled worker')
        w.verify(message,key)
        return payload,'worker:'+route
    if tls_common_name!=claimed:raise ValueError('TLS role binding')
    w.verify(message,peer.verify_keys[claimed])
    return payload,claimed

def serve(peer,handler,port=None):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            msg,_=w.receive(self.request);rid='invalid'
            try:
                names=[v for rdn in self.request.getpeercert()['subject'] for k,v in rdn if k=='commonName']
                if len(names)!=1:raise ValueError('TLS identity missing')
                payload,claimed=authenticate_rpc(peer,msg,names[0]);rid=payload['request_id']
                if payload['task']!=peer.config['task']:raise ValueError('Wrong task')
                with peer.lock:
                    if rid in peer.seen:raise ValueError('Replay request')
                    peer.seen.add(rid)
                    started=time.perf_counter();cpu_started=time.process_time()
                    result={'role':peer.role,'ready':True} if payload['operation']=='health' else handler(payload['operation'],payload['body'],claimed)
                    peer.metrics.append(dict(source=claimed,operation=payload['operation'],handler_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None))
                reply=dict(request_id=rid,ok=True,result=result)
            except Exception as error:
                reply=dict(request_id=rid,ok=False,error=type(error).__name__+': '+str(error))
            w.send(self.request,w.sign(reply,peer.signer))
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address=True;daemon_threads=True;request_queue_size=128
    server=Server(('127.0.0.1',port or peer.config['ports'][peer.role]),Handler)
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.minimum_version=ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(peer.root/peer.role/'tls.pem'),str(peer.root/peer.role/'tls-key.pem'))
    context.load_verify_locations(str(peer.root/'ca.pem'));context.verify_mode=ssl.CERT_REQUIRED
    server.socket=context.wrap_socket(server.socket,server_side=True)
    print(json.dumps({'ready':peer.role,'port':server.server_address[1]}),flush=True)
    server.serve_forever()
