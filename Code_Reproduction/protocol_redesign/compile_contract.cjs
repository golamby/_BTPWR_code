const fs=require('fs');
const path=require('path');
const solc=require(process.argv[2]);
const source=fs.readFileSync(path.join(__dirname,'RequesterRound.sol'),'utf8');
const input={language:'Solidity',sources:{'RequesterRound.sol':{content:source}},settings:{optimizer:{enabled:true,runs:200},evmVersion:'shanghai',outputSelection:{'*':{'*':['abi','evm.bytecode.object']}}}};
const result=JSON.parse(solc.compile(JSON.stringify(input)));
for(const e of result.errors??[])if(e.severity==='error')throw new Error(e.formattedMessage);
fs.writeFileSync(path.join(__dirname,'contract.json'),JSON.stringify({compiler:solc.version(),...result.contracts['RequesterRound.sol'].RequesterRound}));
console.log('compiled RequesterRound');
