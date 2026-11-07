"""Actual EVM execution of the authorized-requester lifecycle contract."""
import json,time
from pathlib import Path
from eth_tester import EthereumTester,PyEVMBackend
from eth_abi import encode,decode
from eth_utils import keccak

class Ledger:
    def __init__(self,artifact_path):
        self.tester=EthereumTester(PyEVMBackend());self.accounts=self.tester.get_accounts()
        self.artifact=json.loads(Path(artifact_path).read_text());self.address=None;self.receipts=[]
    def transact(self,name,types,values,caller):
        data=keccak(text=name+'('+','.join(types)+')')[:4]+encode(types,values)
        return self.raw(data,caller,name)
    def raw(self,data,caller,label,deploy=False):
        tx=dict(from_=self.accounts[caller],gas=10000000,data='0x'+data.hex())
        tx['from']=tx.pop('from_')
        if not deploy:tx['to']=self.address
        start=time.perf_counter();txhash=self.tester.send_transaction(tx)
        receipt=self.tester.get_transaction_receipt(txhash)
        item=dict(label=label,transaction_hash=txhash,gas=receipt['gas_used'],calldata_bytes=len(data),
                  block_number=receipt['block_number'],status=receipt['status'],execution_seconds=time.perf_counter()-start)
        self.receipts.append(item)
        if receipt['status']!=1:raise ValueError('EVM reverted: '+label)
        return receipt
    def deploy(self,genesis,policy):
        if self.address is not None:raise ValueError('Already deployed')
        code=bytes.fromhex(self.artifact['evm']['bytecode']['object'])
        args=encode(['address','address','bytes32','bytes32'],[self.accounts[0],self.accounts[1],bytes.fromhex(genesis),bytes.fromhex(policy)])
        result=self.raw(code+args,0,'deploy',True);self.address=result['contract_address']
        return self.state()
    def view(self,name,kind):
        result=self.tester.call({'from':self.accounts[0],'to':self.address,'data':'0x'+keccak(text=name+'()')[:4].hex()})
        return decode([kind],bytes.fromhex(result[2:]))[0]
    def state(self):
        return dict(address=self.address,epoch=self.view('epoch','uint256'),phase=self.view('phase','uint8'),
            root=self.view('stateRoot','bytes32').hex(),roster=self.view('rosterRoot','bytes32').hex(),
            charge=self.view('charge','uint256'),block=self.tester.get_block_by_number('latest')['number'])
    def dispatch(self,op,b,role):
        if op=='deploy':
            if role!='broker':raise ValueError('Wrong role')
            return self.deploy(b['genesis'],b['policy'])
        if op=='state':return self.state()
        if op=='open':
            if role!='requester':raise ValueError('Wrong role')
            s=self.state();self.transact('open',['uint256','bytes32','bytes32','uint256','uint256'],
                [b['epoch'],bytes.fromhex(b['offers']),bytes.fromhex(b['previous']),b['budget'],s['block']+100000],0)
        elif op=='lock':
            if role!='broker':raise ValueError('Wrong role')
            self.transact('lock',['address[]','uint256[]'],[b['roster'],b['amounts']],1)
        elif op=='settle':
            if role!='requester':raise ValueError('Wrong role')
            self.transact('settle',['uint256']+['bytes32']*5,
                [b['epoch']]+[bytes.fromhex(b[k]) for k in ('previous','roster','reports','scores','next')],0)
        elif op=='receipts':return self.receipts
        else:raise ValueError('Unknown operation')
        return self.state()

def self_test(artifact):
    ledger=Ledger(artifact);root='11'*32;policy='22'*32
    ledger.deploy(root,policy)
    failed=[]
    def reject(label,fn):
        before=ledger.state()
        try:fn()
        except Exception:
            after=ledger.state()
            if any(before[k]!=after[k] for k in ('epoch','phase','root','roster','charge')):raise AssertionError('Revert changed state')
            failed.append(label);return
        raise AssertionError('Accepted negative case '+label)
    reject('wrong requester',lambda:ledger.transact('open',['uint256','bytes32','bytes32','uint256','uint256'],[1,b'a'*32,bytes.fromhex(root),450,1000],2))
    ledger.dispatch('open',dict(epoch=1,offers='33'*32,previous=root,budget=450),'requester')
    reject('duplicate open',lambda:ledger.dispatch('open',dict(epoch=1,offers='33'*32,previous=root,budget=450),'requester'))
    roster=sorted(ledger.accounts[2:5],key=lambda x:int(x,16))
    reject('overspend',lambda:ledger.dispatch('lock',dict(roster=roster,amounts=[200]*3),'broker'))
    reject('duplicate roster',lambda:ledger.dispatch('lock',dict(roster=[roster[0],roster[0]],amounts=[1,1]),'broker'))
    ledger.dispatch('lock',dict(roster=roster,amounts=[100]*3),'broker');s=ledger.state()
    valid=dict(epoch=1,previous=root,roster=s['roster'],reports='44'*32,scores='55'*32,next='66'*32)
    reject('stale root',lambda:ledger.dispatch('settle',dict(valid,previous='77'*32),'requester'))
    reject('wrong roster',lambda:ledger.dispatch('settle',dict(valid,roster='88'*32),'requester'))
    reject('wrong epoch',lambda:ledger.dispatch('settle',dict(valid,epoch=2),'requester'))
    ledger.dispatch('settle',valid,'requester')
    reject('replayed settlement',lambda:ledger.dispatch('settle',valid,'requester'))
    ledger.dispatch('open',dict(epoch=2,offers='33'*32,previous='66'*32,budget=450),'requester')
    ledger.dispatch('lock',dict(roster=roster,amounts=[100]*3),'broker');s=ledger.state()
    reject('reused score digest',lambda:ledger.dispatch('settle',dict(valid,epoch=2,previous='66'*32,roster=s['roster'],next='99'*32),'requester'))
    return dict(negative_cases=len(failed),rejected=failed,receipts=ledger.receipts)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('artifact');p.add_argument('--output',required=True);a=p.parse_args()
    result=self_test(a.artifact);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({'negative_cases':result['negative_cases']}))
