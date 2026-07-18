/* Concrete Compressive Strength - XGBoost regressor reimplemented in pure JS */
/* Mirrors the trained model: XGBRegressor(objective='reg:squarederror', random_state=42) */
/* XGBoost defaults reproduced: n_estimators=100, learning_rate(eta)=0.3, max_depth=6, */
/* reg_lambda=1, gamma=0, min_child_weight=1, base_score=mean(y). */
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

/* Min-max interval range of each input variable (computed from the dataset) */
var RANGES = (function(){
var NF=10; var mn=[], mx=[];
for(var f=0; f<NF; f++){ mn[f]=Infinity; mx[f]=-Infinity; }
for(var i=0;i<DATASET.length;i++){
for(var f=0; f<NF; f++){
var v=DATASET[i].X[f];
if(v<mn[f]) mn[f]=v;
if(v>mx[f]) mx[f]=v;
}
}
var r=[]; for(var f=0; f<NF; f++) r.push([mn[f], mx[f]]);
return r;
})();

var MODEL = (function(){
var NF = 10;
/* ---- XGBoost hyperparameters (defaults used by XGBRegressor) ---- */
var N_ESTIMATORS = 100;
var ETA = 0.3;            /* learning_rate */
var MAX_DEPTH = 6;
var LAMBDA = 1.0;         /* reg_lambda (L2) */
var GAMMA = 0.0;          /* min_split_loss */
var MIN_CHILD_WEIGHT = 1.0;
/* ---------------------------------------------------------------- */
var trees = [];
var base = 0;

function mean(a){ var s=0; for(var i=0;i<a.length;i++) s+=a[i]; return s/a.length; }

/* Leaf weight per XGBoost: w = -sum(g)/(sum(h)+lambda). For squared error h=1. */
function leafWeight(G, H){ return -G/(H+LAMBDA); }

/* Build one regression tree on gradients/hessians (h=1 for reg:squarederror). */
function buildTree(rows, grad, depth){
var node = {};
var G=0, H=0;
for(var i=0;i<rows.length;i++){ G+=grad[rows[i]]; H+=1; }
node.w = ETA * leafWeight(G, H);
if(depth>=MAX_DEPTH || rows.length<=1){ node.leaf=true; return node; }
var parentScore = (G*G)/(H+LAMBDA);
var bestGain=0, bestF=-1, bestT=0, bestL=null, bestR=null;
for(var f=0; f<NF; f++){
var uniq=[]; for(var i=0;i<rows.length;i++) uniq.push(DATASET[rows[i]].X[f]);
uniq.sort(function(a,b){return a-b;});
for(var u=0; u<uniq.length-1; u++){
if(uniq[u]===uniq[u+1]) continue;
var t=(uniq[u]+uniq[u+1])/2;
var L=[],R=[],gL=0,hL=0,gR=0,hR=0;
for(var i=0;i<rows.length;i++){
if(DATASET[rows[i]].X[f]<=t){ L.push(rows[i]); gL+=grad[rows[i]]; hL+=1; }
else { R.push(rows[i]); gR+=grad[rows[i]]; hR+=1; }
}
if(hL<MIN_CHILD_WEIGHT || hR<MIN_CHILD_WEIGHT) continue;
var gain = 0.5*((gL*gL)/(hL+LAMBDA) + (gR*gR)/(hR+LAMBDA) - parentScore) - GAMMA;
if(gain>bestGain){ bestGain=gain; bestF=f; bestT=t; bestL=L; bestR=R; }
}
}
if(bestF<0){ node.leaf=true; return node; }
node.leaf=false; node.f=bestF; node.t=bestT;
node.left=buildTree(bestL, grad, depth+1);
node.right=buildTree(bestR, grad, depth+1);
return node;
}

function predTree(node, x){
while(!node.leaf){ node = (x[node.f]<=node.t)? node.left : node.right; }
return node.w;
}

function train(){
var y=[]; for(var i=0;i<DATASET.length;i++) y.push(DATASET[i].y);
base = mean(y);   /* base_score = mean(y) for regression */
var pred=[]; for(var i=0;i<DATASET.length;i++) pred.push(base);
var rows=[]; for(var i=0;i<DATASET.length;i++) rows.push(i);
for(var m=0;m<N_ESTIMATORS;m++){
/* gradient of 0.5*(pred-y)^2 = pred - y ; hessian = 1 */
var grad=[]; for(var i=0;i<DATASET.length;i++) grad.push(pred[i]-y[i]);
var tree=buildTree(rows, grad, 0);
trees.push(tree);
for(var i=0;i<DATASET.length;i++) pred[i]+= predTree(tree, DATASET[i].X);
}
}

function predict(x){
var p=base;
for(var m=0;m<trees.length;m++) p+= predTree(trees[m], x);
if(p<0) p=0;
return p;
}

train();
return { predict: predict, params: { n_estimators:N_ESTIMATORS, learning_rate:ETA, max_depth:MAX_DEPTH, reg_lambda:LAMBDA, gamma:GAMMA, min_child_weight:MIN_CHILD_WEIGHT, objective:"reg:squarederror", random_state:42 } };
})();
