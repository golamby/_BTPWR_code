// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Authorized-requester model: verifies lifecycle and resource statements,
// not the numerical correctness of requester recovery or scoring.
contract RequesterRound {
    address public immutable requester;
    address public immutable matcher;
    bytes32 public immutable policy;
    bytes32 public stateRoot;
    bytes32 public offerRoot;
    bytes32 public rosterRoot;
    uint256 public epoch;
    uint256 public budget;
    uint256 public charge;
    uint256 public deadline;
    uint8 public phase;
    uint256 public recipientCount;
    mapping(bytes32=>bool) public usedScoreRoots;
    event Opened(uint256 indexed epoch,bytes32 offers,bytes32 previous,uint256 cap);
    event Locked(uint256 indexed epoch,bytes32 roster,uint256 amount,uint256 recipients);
    event Advanced(uint256 indexed epoch,bytes32 reports,bytes32 scores,bytes32 next);

    constructor(address r,address m,bytes32 genesis,bytes32 rules) {
        require(r!=address(0)&&m!=address(0)&&r!=m&&genesis!=0&&rules!=0,"init");
        requester=r;matcher=m;stateRoot=genesis;policy=rules;
    }
    function open(uint256 nextEpoch,bytes32 offers,bytes32 previous,uint256 cap,uint256 expires) external {
        require(msg.sender==requester,"requester");
        require(phase==0&&nextEpoch==epoch+1&&previous==stateRoot,"freshness");
        require(offers!=0&&cap>0&&expires>block.number,"context");
        epoch=nextEpoch;offerRoot=offers;budget=cap;deadline=expires;phase=1;
        emit Opened(epoch,offers,previous,cap);
    }
    function lock(address[] calldata roster,uint256[] calldata amounts) external {
        require(msg.sender==matcher&&phase==1&&block.number<=deadline,"match");
        require(roster.length==amounts.length&&roster.length<=100,"size");
        uint256 total;
        for(uint256 i=0;i<roster.length;i++) {
            require(roster[i]!=address(0)&&(i==0||roster[i-1]<roster[i]),"roster");
            total+=amounts[i];
        }
        require(total<=budget,"budget");
        rosterRoot=keccak256(abi.encode(roster,amounts));charge=total;
        recipientCount=roster.length;phase=2;
        emit Locked(epoch,rosterRoot,total,roster.length);
    }
    function settle(uint256 currentEpoch,bytes32 previous,bytes32 roster,bytes32 reports,bytes32 scores,bytes32 next) external {
        require(msg.sender==requester&&phase==2&&block.number<=deadline,"settle");
        require(currentEpoch==epoch&&previous==stateRoot&&roster==rosterRoot,"binding");
        require(reports!=0&&scores!=0&&next!=0&&next!=stateRoot,"roots");
        require(!usedScoreRoots[scores],"score replay");
        usedScoreRoots[scores]=true;stateRoot=next;phase=0;
        emit Advanced(epoch,reports,scores,next);
    }
    function expire() external {
        require(phase!=0&&block.number>deadline,"deadline");
        // A failed round does not issue a replacement reputation state.
        phase=0;
    }
}
