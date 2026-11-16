import subprocess, os, glob, json, random, collections
from scipy import stats
D=os.path.expanduser("~/ir-ca1"); T=D+"/trec_eval/trec_eval"; Q=D+"/qrels.txt"
RUNS=D+"/runs-prf"; CACHE=D+"/prf_ap.json"
mean=lambda xs: sum(xs)/len(xs)
def te(args,run): return subprocess.run([T]+args+[Q,run],capture_output=True,text=True).stdout
def per_query(run):
    return {int(l.split()[1]):float(l.split()[2]) for l in te(["-q","-m","map"],run).splitlines() if len(l.split())==3 and l.split()[1]!="all"}
def full(run):
    return {l.split()[0]:float(l.split()[2]) for l in te(["-m","map","-m","recall.10,100","-m","ndcg_cut.10","-m","P.10"],run).splitlines() if len(l.split())==3}
def row(name,run):
    m=full(run); return f"| {name} | {m['map']:.4f} | {m['recall_10']:.4f} | {m['recall_100']:.4f} | {m['P_10']:.4f} | {m['ndcg_cut_10']:.4f} |"
if os.path.exists(CACHE):
    ap={k:{int(q):v for q,v in d.items()} for k,d in json.load(open(CACHE)).items()}
else:
    ap={os.path.basename(f)[:-4]:per_query(f) for f in sorted(glob.glob(RUNS+"/*.run"))}
    json.dump(ap,open(CACHE,"w"))
qs=[];inw=False
for l in open(D+"/data/cran.qry",encoding="latin-1"):
    if l.startswith(".I"): qs.append(""); inw=False
    elif l.startswith(".W"): inw=True
    elif inw: qs[-1]+=l.strip()+" "
qids=sorted(ap["def_base"]); rnd=random.Random(1); sh=qids[:]; rnd.shuffle(sh)
folds=[sh[i::5] for i in range(5)]
print("| System | MAP | R@10 | R@100 | P@10 | nDCG@10 |"); print("|---|---|---|---|---|---|")
rows=[]; modes={}
for tag,label in [("def","BM25 default"),("tun","BM25 tuned (4.0, 0.9)")]:
    cfgs=[n for n in ap if n.startswith(tag+"_N")]; base=ap[tag+"_base"]
    cv={}; chosen=[]
    for f in folds:
        tr=[q for q in qids if q not in f]
        b=max(cfgs,key=lambda n: mean([ap[n][q] for q in tr])); chosen.append(b)
        for q in f: cv[q]=ap[b][q]
    mode=collections.Counter(chosen).most_common(1)[0][0]; modes[tag]=mode
    x=[cv[q] for q in qids]; y=[base[q] for q in qids]
    orc=max(cfgs,key=lambda n: mean(list(ap[n].values())))
    try: wp=stats.wilcoxon(x,y).pvalue
    except Exception: wp=float("nan")
    rows.append(f"{label}: no-PRF MAP={mean(y):.4f} | PRF 5-fold CV MAP={mean(x):.4f} (diff {mean(x)-mean(y):+.4f}, t-test p={stats.ttest_rel(x,y).pvalue:.4f}, Wilcoxon p={wp:.4f}) | queries better/worse/tie={sum(a>b+1e-9 for a,b in zip(x,y))}/{sum(a<b-1e-9 for a,b in zip(x,y))}/{sum(abs(a-b)<=1e-9 for a,b in zip(x,y))} | in-sample best={orc} MAP={mean(list(ap[orc].values())):.4f} | chosen per fold={chosen}")
    print(row(label+" (no PRF)",f"{RUNS}/{tag}_base.run"))
    print(row(label+f" + PRF [{mode}]",f"{RUNS}/{mode}.run"))
print("\nCross-validation summary:"); [print(" ",r) for r in rows]
m=modes["def"]; b=ap["def_base"]; d=sorted(qids,key=lambda q: ap[m][q]-b[q])
print(f"\nPer-query examples ({m} vs no PRF):")
for q in d[-4:][::-1]+d[:4]: print(f"  q{q} base={b[q]:.3f} prf={ap[m][q]:.3f} | {qs[q-1][:95]}")
import collections as C
def load(f):
    r=C.defaultdict(list)
    for l in open(f):
        q,_,dd,rk,_,_=l.split(); r[q].append((int(rk),dd))
    return r
def rrf(files,out,k=60):
    runs=[load(f) for f in files]
    with open(out,"w") as w:
        for q in runs[0]:
            s=C.defaultdict(float)
            for r in runs:
                for rk,dd in r[q]: s[dd]+=1.0/(k+rk)
            for i,(dd,v) in enumerate(sorted(s.items(),key=lambda z:-z[1])[:1000]): w.write(f"{q} Q0 {dd} {i+1} {v:.6f} rrf\n")
F=D+"/final/"; os.makedirs(D+"/fused",exist_ok=True)
print("\nRank fusion (RRF, k=60):"); print("| System | MAP | R@10 | R@100 | P@10 | nDCG@10 |"); print("|---|---|---|---|---|---|")
for i,(n,fs) in enumerate({"RRF: English BM25 + VSM + LMD":[F+"english_bm25.run",F+"english_vsm.run",F+"english_lmd_50.run"],
    "RRF: tuned BM25 + VSM + LMD":[F+"english_bm25_4.0_0.9.run",F+"english_vsm.run",F+"english_lmd_50.run"],
    "RRF: PRF(tuned) + BM25 tuned + VSM":[f"{RUNS}/{modes['tun']}.run",F+"english_bm25_4.0_0.9.run",F+"english_vsm.run"],
    "RRF: standard + english + custom BM25":[F+"standard_bm25.run",F+"english_bm25.run",F+"custom_bm25.run"]}.items()):
    o=f"{D}/fused/f{i}.run"; rrf(fs,o); print(row(n,o))
