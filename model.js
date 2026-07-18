/* Concrete Compressive Strength - XGBoost-style gradient boosting (pure JS, trains in-browser) */
/* Feature order: [cement, nca, rca, sf, ggbfs, lp, np, sp, slump, age] */
var DATASET = [
{name:"OAC-1", X:[380,1150,0,0,0,0,0,1.0,80,7], y:32.5},
{name:"OAC-1", X:[380,1150,0,0,0,0,0,1.0,80,28], y:44.1},
{name:"OAC-1", X:[380,1150,0,0,0,0,0,1.0,80,90], y:50.3},
{name:"OAC-2", X:[420,1100,0,30,0,0,0,1.5,70,7], y:38.2},
{name:"OAC-2", X:[420,1100,0,30,0,0,0,1.5,70,28], y:52.6},
{name:"OAC-2", X:[420,1100,0,30,0,0,0,1.5,70,90], y:60.4},
{name:"RAC-30", X:[380,805,345,0,0,0,0,1.2,75,7], y:28.9},
{name:"RAC-30", X:[380,805,345,0,0,0,0,1.2,75,28], y:39.7},
{name:"RAC-30", X:[380,805,345,0,0,0,0,1.2,75,90], y:45.2},
{name:"RAC-50", X:[360,575,575,0,0,0,0,1.3,70,7], y:24.6},
{name:"RAC-50", X:[360,575,575,0,0,0,0,1.3,70,28], y:34.8},
{name:"RAC-50", X:[360,575,575,0,0,0,0,1.3,70,90], y:40.1},
{name:"RAC-100", X:[350,0,1100,0,0,0,0,1.6,65,7], y:19.8},
{name:"RAC-100", X:[350,0,1100,0,0,0,0,1.6,65,28], y:28.5},
{name:"RAC-100", X:[350,0,1100,0,0,0,0,1.6,65,90], y:33.7},
{name:"SCM-GGBFS", X:[300,1050,0,0,120,0,0,1.4,90,28], y:47.9},
{name:"SCM-GGBFS", X:[300,1050,0,0,120,0,0,1.4,90,90], y:58.2},
{name:"SCM-LP", X:[340,1080,0,0,0,60,0,1.1,85,28], y:41.3},
{name:"SCM-LP", X:[340,1080,0,0,0,60,0,1.1,85,90], y:46.8},
{name:"SCM-NP", X:[320,1060,0,0,0,0,70,1.2,80,28], y:39.5},
{name:"SCM-NP", X:[320,1060,0,0,0,0,70,1.2,80,90], y:45.9},
{name:"RAC50-SF", X:[400,560,560,25,0,0,0,1.8,60,28], y:45.6},
{name:"RAC50-SF", X:[400,560,560,25,0,0,0,1.8,60,90], y:53.1},
{name:"HighEarly", X:[450,1080,0,20,60,0,0,2.0,65,7], y:42.7}
];

var MODEL = (function(){
var NF = 10;
var trees = [];
var base = 0;
var lr = 0.15;
var nEstimators = 120;
var maxDepth = 3;
var minLeaf = 2;
var lambda = 1.0;

function mean(a){ var s=0; for(var i=0;i<a.length;i++) s+=a[i]; return s/a.length; }

function buildTree(rows, grad, depth){
var node = {};
var idx = rows;
var g = []; for(var i=0;i<idx.length;i++) g.push(grad[idx[i]]);
var leafVal = mean(g);
if(depth>=maxDepth || idx.length<=minLeaf){ node.leaf=true; node.val=leafVal; return node; }
var bestGain=-Infinity, bestF=-1, bestT=0, bestL=null, bestR=null;
for(var f=0; f<NF; f++){
var vals=[]; for(var i=0;i<idx.length;i++) vals.push(DATASET[idx[i]].X[f]);
var uniq = vals.slice().sort(function(a,b){return a-b;});
for(var u=0; u<uniq.length-1; u++){
var t=(uniq[u]+uniq[u+1])/2;
var L=[],R=[];
for(var i=0;i<idx.length;i++){ if(DATASET[idx[i]].X[f]<=t) L.push(idx[i]); else R.push(idx[i]); }
if(L.length===0||R.length===0) continue;
var gL=0,gR=0; for(var i=0;i<L.length;i++) gL+=grad[L[i]]; for(var i=0;i<R.length;i++) gR+=grad[R[i]];
var gain = (gL*gL)/(L.length+lambda) + (gR*gR)/(R.length+lambda);
if(gain>bestGain){ bestGain=gain; bestF=f; bestT=t; bestL=L; bestR=R; }
}
}
if(bestF<0){ node.leaf=true; node.val=leafVal; return node; }
node.leaf=false; node.f=bestF; node.t=bestT;
node.left=buildTree(bestL, grad, depth+1);
node.right=buildTree(bestR, grad, depth+1);
return node;
}

function predTree(node, x){
while(!node.leaf){ node = (x[node.f]<=node.t)? node.left : node.right; }
return node.val;
}

function train(){
var y=[]; for(var i=0;i<DATASET.length;i++) y.push(DATASET[i].y);
base = mean(y);
var pred=[]; for(var i=0;i<DATASET.length;i++) pred.push(base);
var allRows=[]; for(var i=0;i<DATASET.length;i++) allRows.push(i);
for(var m=0;m<nEstimators;m++){
var grad=[]; for(var i=0;i<DATASET.length;i++) grad.push(y[i]-pred[i]);
var tree=buildTree(allRows, grad, 0);
trees.push(tree);
for(var i=0;i<DATASET.length;i++) pred[i]+= lr*predTree(tree, DATASET[i].X);
}
}

function predict(x){
var p=base;
for(var m=0;m<trees.length;m++) p+= lr*predTree(trees[m], x);
if(p<0) p=0;
return p;
}

train();
return { predict: predict };
})();
