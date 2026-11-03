import subprocess, sys, glob, os, re, csv

trec  = os.path.expanduser("~/ir-ca1/trec_eval/trec_eval")
qrels = os.path.expanduser("~/ir-ca1/qrels.txt")
rundir, out = sys.argv[1], sys.argv[2]
delete = len(sys.argv) > 3 and sys.argv[3] == "--delete"
SPLIT = 112                      # queries 1..112 = train, 113..225 = test

def mean(xs): return sum(xs) / len(xs) if xs else 0.0

rows = []
for f in sorted(glob.glob(rundir + "/*.run")):
    m = re.search(r"k([\d.]+)_b([\d.]+)\.run$", f)
    k1, b = float(m.group(1)), float(m.group(2))
    res = subprocess.run([trec, "-q", "-m", "map", qrels, f],
                         capture_output=True, text=True).stdout
    ap = {}
    for line in res.splitlines():
        p = line.split()
        if len(p) == 3 and p[1] != "all":
            ap[int(p[1])] = float(p[2])
    rows.append((k1, b,
                 mean(list(ap.values())),
                 mean([v for q, v in ap.items() if q <= SPLIT]),
                 mean([v for q, v in ap.items() if q > SPLIT])))
    if delete: os.remove(f)

with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["k1", "b", "map_all", "map_train", "map_test"])
    w.writerows(rows)

best_all   = max(rows, key=lambda r: r[2])
best_train = max(rows, key=lambda r: r[3])
print(f"Best on ALL queries : k1={best_all[0]}, b={best_all[1]}, MAP={best_all[2]:.4f}")
print(f"Best on TRAIN       : k1={best_train[0]}, b={best_train[1]}, "
      f"train MAP={best_train[3]:.4f}, held-out TEST MAP={best_train[4]:.4f}")
python3 ~/ir-ca1/eval_grid.py ~/ir-ca1/runs-grid ~/ir-ca1/grid_english.csv --delete
