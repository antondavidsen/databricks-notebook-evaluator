# Databricks notebook source
# MAGIC %md
# MAGIC ## cash app v3
# MAGIC new logic from Q2 - dont use v2 notebook anymore

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import re, json, time, datetime, math
from functools import reduce

# COMMAND ----------

dbutils.widgets.text("src", "/mnt/fin-raw/cashapp/")
dbutils.widgets.text("run_dt", "")
dbutils.widgets.dropdown("co", "ALL", ["ALL", "DK01", "IN02", "SG01", "US01"])
dbutils.widgets.dropdown("dry", "Y", ["Y", "N"])

src = dbutils.widgets.get("src")
run_dt = dbutils.widgets.get("run_dt")
if run_dt == "":
    run_dt = str(datetime.date.today())
co_f = dbutils.widgets.get("co")
dry = dbutils.widgets.get("dry")
print(src, run_dt, co_f, dry)

# COMMAND ----------

TOL = 0.5
MAX_WO = 250.0
EX_TOL = 0.01
bk = {"DB-4471": "DK01", "HDFC-0932": "IN02", "DBS-1180": "SG01", "JPM-2290": "US01"}
EXCL = ["C10077", "C10412"]
tgt = "fin_cash.alloc_lines"
t0 = time.time()
spark.conf.set("spark.sql.shuffle.partitions", 8)

# COMMAND ----------

# MAGIC %run ../shared/fin_utils

# COMMAND ----------

r = spark.read.option("header", True).csv(src + "bank_receipts.csv")
r = r.withColumn("amt", F.regexp_replace("amt", ",", "").cast("double"))
r = r.withColumn("val_dt", F.coalesce(F.to_date("val_dt", "yyyy-MM-dd"), F.to_date("val_dt", "dd/MM/yyyy")))
r = r.withColumn("ccy", F.upper(F.trim("ccy")))
print(r.count())
# display(r)

# COMMAND ----------

r = r.filter(F.col("amt") > 0).filter(F.col("val_dt") <= F.lit(run_dt))
m = F.create_map([F.lit(x) for kv in bk.items() for x in kv])
r = r.withColumn("co", m[F.col("bank_acct")])
r = r.filter(F.col("co").isNotNull())
if co_f != "ALL":
    r = r.filter(F.col("co") == co_f)
print(r.count())

# COMMAND ----------

oi = spark.read.option("header", True).csv(src + "open_items.csv")
oi = oi.withColumn("open_amt", F.col("open_amt").cast("double")) \
    .withColumn("doc_dt", F.to_date("doc_dt")) \
    .withColumn("due_dt", F.to_date("due_dt"))
oi = oi.filter("open_amt > 0")
oi = oi.filter(~F.col("cust_no").isin(EXCL))
if co_f != "ALL":
    oi = oi.filter(F.col("co_cd") == co_f)
oi = oi.withColumn("dispute", F.when(F.col("dispute") == "Y", 1).otherwise(0))
print(oi.count(), oi.filter("dispute=1").count())

# COMMAND ----------

cm = spark.read.option("header", True).csv(src + "customers.csv")
cm = cm.withColumn("pool", F.coalesce(F.col("parent_no"), F.col("cust_no")))
cm.cache()
print(cm.count())

# COMMAND ----------

def nrm(s):
    if s is None:
        return None
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    for w in [" PVT", " LTD", " LIMITED", " PTE", " INC", " LLC", " GMBH", " A S", " CO"]:
        s = s.replace(w, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

nrm_u = F.udf(nrm, StringType())

def get_refs(t):
    if not t:
        return []
    return re.findall(r"(?<!\d)(9\d{7})(?!\d)", t)

refs_u = F.udf(get_refs, ArrayType(StringType()))

# COMMAND ----------

al = cm.select("cust_no", "pool", F.explode(F.split("alias", "\\|")).alias("a"))
al = al.withColumn("a", nrm_u("a")).dropna(subset=["a"]).dropDuplicates(["cust_no", "a"])
al2 = cm.select("cust_no", "pool", nrm_u("cust_nm").alias("a"))
al = al.unionByName(al2).dropDuplicates(["cust_no", "a"])
print(al.count())
# display(al.groupBy("a").count().filter("count>1"))

# COMMAND ----------

r = r.withColumn("refs", refs_u("remit_txt"))
r = r.withColumn("pn", nrm_u("payer_nm"))
r.cache()
display(r.select("rcpt_id", "payer_nm", "pn", "remit_txt", "refs").limit(20))

# COMMAND ----------

x1 = r.select("rcpt_id", F.explode("refs").alias("doc_no")) \
    .join(oi.select("doc_no", "cust_no"), "doc_no") \
    .groupBy("rcpt_id").agg(F.first("cust_no").alias("c_ref"))

x2 = r.join(al, r.pn == al.a, "inner").select("rcpt_id", F.col("cust_no").alias("c_nm"))
x2 = x2.dropDuplicates(["rcpt_id"])

ov = spark.read.option("header", True).csv(src + "ar_overrides.csv")
ov = ov.select("rcpt_id", F.col("cust_no").alias("c_ov"))

# COMMAND ----------

rr = r.join(ov, "rcpt_id", "left").join(x1, "rcpt_id", "left").join(x2, "rcpt_id", "left")
rr = rr.withColumn("cust", F.coalesce("c_ov", "c_ref", "c_nm"))
rr = rr.withColumn("id_src", F.when(F.col("c_ov").isNotNull(), "OV")
                   .when(F.col("c_ref").isNotNull(), "REF")
                   .when(F.col("c_nm").isNotNull(), "NM")
                   .otherwise(None))
rr = rr.join(cm.select(F.col("cust_no").alias("cust"), "pool"), "cust", "left")
print(rr.count())
display(rr.groupBy("id_src").count())

# COMMAND ----------

# old v2 matching, keep for now
# ex = rr.join(oi, (rr.cust == oi.cust_no) & (F.abs(rr.amt - oi.open_amt) < 1), "inner")
# ex = ex.withColumn("rn", F.row_number().over(Window.partitionBy("rcpt_id").orderBy("due_dt")))
# ex = ex.filter("rn = 1")
# ex = ex.select("rcpt_id", "doc_no", "amt")
# ex.createOrReplaceTempView("v_exact")
# rest = rr.join(ex, "rcpt_id", "left_anti")
# print(ex.count(), rest.count())
#
# fifo = rest.join(oi, rest.cust == oi.cust_no, "inner")
# fifo = fifo.withColumn("cum", F.sum("open_amt").over(Window.partitionBy("rcpt_id").orderBy("due_dt").rowsBetween(Window.unboundedPreceding, 0)))
# fifo = fifo.withColumn("prev", F.col("cum") - F.col("open_amt"))
# fifo = fifo.filter(F.col("prev") < F.col("amt"))
# fifo = fifo.withColumn("alloc", F.least(F.col("open_amt"), F.col("amt") - F.col("prev")))
# display(fifo)
#
# out_v2 = ex.withColumn("typ", F.lit("EXACT")).unionByName(fifo.select("rcpt_id", "doc_no", F.col("alloc").alias("amt")).withColumn("typ", F.lit("FIFO")))
# out_v2.write.mode("append").saveAsTable("fin_cash.alloc_lines_v2")

# COMMAND ----------

fx = spark.read.option("header", True).csv(src + "fx_rates.csv")
fx = fx.withColumn("usd_rate", F.col("usd_rate").cast("double")).withColumn("rate_dt", F.to_date("rate_dt"))
fx = fx.withColumn("ccy", F.upper("ccy"))
w = Window.partitionBy("ccy").orderBy(F.col("rate_dt").desc())
fx1 = fx.withColumn("rn", F.row_number().over(w)).filter("rn = 1").select("ccy", "usd_rate")
display(fx1)

# COMMAND ----------

fx2 = rr.select("rcpt_id", "ccy", "val_dt").join(fx, "ccy", "left").filter(F.col("rate_dt") <= F.col("val_dt"))
fx2 = fx2.withColumn("rn", F.row_number().over(Window.partitionBy("rcpt_id").orderBy(F.col("rate_dt").desc())))
fx2 = fx2.filter("rn = 1").select("rcpt_id", F.col("usd_rate").alias("vd_rate"))
print(fx2.count())

# COMMAND ----------

rr = rr.join(fx1.withColumnRenamed("usd_rate", "r_rate"), "ccy", "left")
# rr = rr.join(fx2, "rcpt_id", "left").withColumn("r_rate", F.coalesce("vd_rate", "r_rate"))
rr = rr.withColumn("amt_usd", F.col("amt") * F.col("r_rate"))

oi2 = oi.join(cm.select("cust_no", "pool"), "cust_no", "left")
oi2 = oi2.join(fx1.withColumnRenamed("ccy", "doc_ccy").withColumnRenamed("usd_rate", "i_rate"), "doc_ccy", "left")
oi2 = oi2.withColumn("open_usd", F.col("open_amt") * F.col("i_rate"))
print(rr.filter("r_rate is null").count(), oi2.filter("i_rate is null").count())

# COMMAND ----------

dq = {}
dq["rcpt_dup"] = rr.groupBy("rcpt_id").count().filter("count > 1").count()
dq["rcpt_no_dt"] = r.filter("val_dt is null").count()
dq["rcpt_no_cust"] = rr.filter("cust is null").count()
dq["inv_disp"] = oi.filter("dispute = 1").count()
dq["inv_no_pool"] = oi2.filter("pool is null").count()
dq["multi_alias"] = al.groupBy("a").agg(F.countDistinct("cust_no").alias("n")).filter("n > 1").count()
print(json.dumps(dq, indent=1))
if dq["rcpt_dup"] > 0:
    print("WARNING dup receipts!!")

# COMMAND ----------

# import difflib
# def fz(a, b):
#     return difflib.SequenceMatcher(None, a, b).ratio()
# pn_list = [z.pn for z in r.select("pn").distinct().collect()]
# al_list = [(z.a, z.cust_no) for z in al.collect()]
# hits = []
# for p in pn_list:
#     if p is None:
#         continue
#     best = None
#     bs = 0
#     for a, c in al_list:
#         sc = fz(p, a)
#         if sc > bs:
#             bs = sc
#             best = c
#     if bs > 0.86:
#         hits.append((p, best, bs))
# print(len(hits))
# fzdf = spark.createDataFrame(hits, ["pn", "c_fz", "score"])
# display(fzdf.orderBy("score"))
# too many false positives on the INDIA / KENYA ones, parked

# COMMAND ----------

display(rr.groupBy("co", "ccy").agg(F.count("*").alias("n"), F.round(F.sum("amt"), 2).alias("amt"), F.round(F.sum("amt_usd"), 2).alias("usd")).orderBy("co"))

# COMMAND ----------

display(oi2.groupBy("co_cd", "doc_ccy").agg(F.count("*").alias("n"), F.round(F.sum("open_amt"), 2).alias("open"), F.round(F.sum("open_usd"), 2).alias("usd")).orderBy("co_cd"))

# COMMAND ----------

def alloc_by_aging(pdf_inv, amt, asof, buckets=[30, 60, 90]):
    p = pdf_inv.copy()
    p["age"] = (pd.to_datetime(asof) - pd.to_datetime(p["due_dt"])).dt.days
    p["bkt"] = 0
    for i, b in enumerate(buckets):
        p.loc[p["age"] > b, "bkt"] = i + 1
    p = p.sort_values(["bkt", "age"], ascending=[False, False])
    res = []
    left = amt
    for _, q in p.iterrows():
        if left <= 0:
            break
        if q["dispute"] == 1:
            continue
        a = min(left, q["open_usd"])
        res.append((q["doc_no"], a, q["bkt"]))
        left = left - a
    if left > 0:
        res.append((None, left, -1))
    return res

def aging_rpt(pdf_inv, asof):
    p = pdf_inv.copy()
    p["age"] = (pd.to_datetime(asof) - pd.to_datetime(p["due_dt"])).dt.days
    p["bkt"] = pd.cut(p["age"], [-9999, 0, 30, 60, 90, 9999], labels=["cur", "1-30", "31-60", "61-90", "90+"])
    return p.pivot_table(index="pool", columns="bkt", values="open_usd", aggfunc="sum", fill_value=0)

# COMMAND ----------

rp = rr.select("rcpt_id", "val_dt", "amt", "ccy", "r_rate", "amt_usd", "cust", "pool", "refs", "co", "id_src", "payer_nm").toPandas()
ip = oi2.select("doc_no", "cust_no", "pool", "co_cd", "doc_dt", "due_dt", "open_amt", "doc_ccy", "i_rate", "open_usd", "dispute").toPandas()
print(len(rp), len(ip))

# COMMAND ----------

bal = dict(zip(ip.doc_no, ip.open_usd))
meta = ip.set_index("doc_no").to_dict("index")
pools = {}
for d, pl in zip(ip.doc_no, ip.pool):
    pools.setdefault(pl, []).append(d)
print(len(bal), len(pools))

# COMMAND ----------

def apply_ref(rid, rem, refs, lines):
    if refs is None:
        return rem
    for d in refs:
        if rem <= 0:
            break
        if d not in bal:
            continue
        if bal[d] <= 0:
            continue
        a = min(rem, bal[d])
        bal[d] = bal[d] - a
        rem = rem - a
        lines.append([rid, "REF", d, a])
    return rem

def apply_exact(rid, rem, pool, lines):
    if pool not in pools:
        return rem
    c = [d for d in pools[pool] if meta[d]["dispute"] == 0 and bal[d] > 0 and abs(bal[d] - rem) <= EX_TOL]
    if len(c) == 0:
        return rem
    c = sorted(c, key=lambda d: meta[d]["doc_dt"])
    d = c[0]
    a = min(rem, bal[d])
    bal[d] = bal[d] - a
    rem = rem - a
    lines.append([rid, "EXACT", d, a])
    return rem

def apply_fifo(rid, rem, pool, lines):
    if pool not in pools:
        return rem
    ds = [d for d in pools[pool] if meta[d]["dispute"] == 0 and bal[d] > 0]
    ds = sorted(ds, key=lambda d: -bal[d])
    for d in ds:
        if rem <= 0.005:
            break
        a = min(rem, bal[d])
        bal[d] = bal[d] - a
        rem = rem - a
        lines.append([rid, "FIFO", d, a])
    return rem

def do_wo(rid, lines):
    t = set([l[2] for l in lines if l[0] == rid and l[1] in ("REF", "EXACT", "FIFO")])
    for d in sorted(t):
        if bal[d] > 0.005 and bal[d] <= TOL:
            lines.append([rid, "WO", d, bal[d]])
            bal[d] = 0.0

def check_bal():
    neg = [d for d, v in bal.items() if v < -0.005]
    if len(neg) > 0:
        print("NEG BAL", neg[:10])
    return len(neg)

# COMMAND ----------

tmp = rp[rp.id_src.isnull()]
print(len(tmp))
display(spark.createDataFrame(tmp[["rcpt_id", "payer_nm", "amt", "ccy"]]))

# COMMAND ----------

# display(rr.filter("rcpt_id = 'R-10116'"))
# display(oi2.filter("pool = 'C10011'"))
x = rp[rp.cust.notnull()].groupby("pool").amt_usd.sum().sort_values(ascending=False).head(10)
print(x)

# COMMAND ----------

tot_in = rp.amt_usd.sum()
tot_open = ip.open_usd.sum()
print(round(tot_in, 2), round(tot_open, 2), round(tot_in / tot_open, 3))
TOL = 5.0
print(len(rp[rp.refs.apply(lambda z: z is not None and len(z) > 0)]))

# COMMAND ----------

lines = []
rp = rp.sort_values(["val_dt", "rcpt_id"]).reset_index(drop=True)
for i, x in rp.iterrows():
    rid = x.rcpt_id
    rem = x.amt_usd
    n0 = len(lines)
    rem = apply_ref(rid, rem, x.refs, lines)
    if len(lines) == n0 and pd.notnull(x.cust):
        rem = apply_exact(rid, rem, x.pool, lines)
    if rem > TOL and pd.notnull(x.cust):
        rem = apply_fifo(rid, rem, x.pool, lines)
    do_wo(rid, lines)
    if rem > 0.005:
        if pd.notnull(x.cust):
            lines.append([rid, "UNAPP", None, rem])
        else:
            lines.append([rid, "SUSP", None, rem])
print(len(lines), check_bal())

# COMMAND ----------

ln = pd.DataFrame(lines, columns=["rcpt_id", "typ", "doc_no", "amt_usd"])
ln = ln.merge(rp[["rcpt_id", "ccy", "r_rate", "amt", "co", "cust", "val_dt"]], on="rcpt_id", how="left")
ln = ln.merge(ip[["doc_no", "doc_ccy", "i_rate", "cust_no", "co_cd"]], on="doc_no", how="left")
ln["amt_rc"] = (ln.amt_usd / ln.r_rate).round(2)
ln["amt_dc"] = (ln.amt_usd / ln.i_rate).round(2)
ln["ln"] = ln.groupby("rcpt_id").cumcount() + 1
print(ln.typ.value_counts())

# COMMAND ----------

s = ln[ln.typ != "WO"].groupby("rcpt_id").amt_rc.sum()
d = (ln.drop_duplicates("rcpt_id").set_index("rcpt_id").amt - s).round(2)
d = d[d != 0]
print(len(d), d.abs().max() if len(d) else 0)
for rid, v in d.items():
    j = ln[(ln.rcpt_id == rid) & (ln.typ != "WO")].index[-1]
    ln.loc[j, "amt_rc"] = round(ln.loc[j, "amt_rc"] + v, 2)
    if ln.loc[j, "doc_ccy"] == ln.loc[j, "ccy"]:
        ln.loc[j, "amt_dc"] = round(ln.loc[j, "amt_dc"] + v, 2)

# COMMAND ----------

gm = {}
for c in ln.co.dropna().unique():
    gm[c] = get_gl_map(c)
def _gl(z):
    k = "AR" if z.typ in ("REF", "EXACT", "FIFO") else z.typ
    return gm[z.co][k]
ln["gl"] = ln.apply(_gl, axis=1)
ln["run_dt"] = run_dt
print(ln.head())

# COMMAND ----------

chk = ln[ln.typ != "WO"].groupby("rcpt_id").amt_rc.sum().round(2)
chk = pd.concat([chk, rp.set_index("rcpt_id").amt], axis=1)
chk.columns = ["alloc", "amt"]
bad = chk[(chk.alloc - chk.amt).abs() > 0.001]
print("recon mismatches:", len(bad))
if len(bad) > 0:
    print(bad.head(20))
smry = ln.groupby("typ").agg(n=("rcpt_id", "count"), usd=("amt_usd", "sum")).round(2)
print(smry)

# COMMAND ----------

st = ln.merge(rp[["rcpt_id", "id_src"]], on="rcpt_id", how="left")
st["id_src"] = st.id_src.fillna("NONE")
hr = st.groupby(["id_src", "typ"]).amt_usd.sum().unstack(fill_value=0).round(0)
print(hr)
auto = ln[ln.typ.isin(["REF", "EXACT", "FIFO", "WO"])].rcpt_id.nunique()
full = rp.rcpt_id.nunique() - ln[ln.typ.isin(["UNAPP", "SUSP"])].rcpt_id.nunique()
print("auto touched", auto, "of", len(rp), "| fully applied", full, "| rate", round(full / len(rp) * 100, 1))

# COMMAND ----------

sch = StructType([
    StructField("rcpt_id", StringType()), StructField("ln", IntegerType()), StructField("typ", StringType()),
    StructField("doc_no", StringType()), StructField("cust_no", StringType()), StructField("co", StringType()),
    StructField("ccy", StringType()), StructField("amt_rc", DoubleType()), StructField("doc_ccy", StringType()),
    StructField("amt_dc", DoubleType()), StructField("amt_usd", DoubleType()), StructField("gl", StringType()),
    StructField("run_dt", StringType())])
ln["cust_no"] = ln.cust_no.fillna(ln.cust)
o = ln[["rcpt_id", "ln", "typ", "doc_no", "cust_no", "co", "ccy", "amt_rc", "doc_ccy", "amt_dc", "amt_usd", "gl", "run_dt"]]
o = o.astype(object).where(o.notnull(), None)
out = spark.createDataFrame(o.values.tolist(), sch)
out = out.withColumn("ld_ts", F.current_timestamp())
display(out)

# COMMAND ----------

if dry == "N":
    out.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(tgt)
    print("written", out.count())
else:
    print("dry run, not written")

# COMMAND ----------

ip["open_usd_after"] = ip.doc_no.map(bal)
ip["open_after"] = (ip.open_usd_after / ip.i_rate).round(2)
ip["open_after"] = np.where(ip.open_usd_after.abs() < 0.005, 0.0, ip.open_after)
ip["status"] = np.where(ip.open_after == 0, "CLEARED", np.where(ip.open_after < ip.open_amt, "PARTIAL", "OPEN"))
print(ip.status.value_counts())
oa = spark.createDataFrame(ip[["doc_no", "cust_no", "co_cd", "doc_ccy", "open_amt", "open_after", "status"]].astype({"open_amt": float, "open_after": float}))
oa = oa.withColumn("run_dt", F.lit(run_dt))
if dry == "N":
    oa.write.format("delta").mode("overwrite").saveAsTable("fin_cash.open_items_after")

# COMMAND ----------

ag = ip.copy()
ag["open_usd"] = ag.open_usd_after
ag = ag[ag.open_usd > 0.005]
snap = aging_rpt(ag, run_dt).reset_index()
snap.columns = [str(c) for c in snap.columns]
snap["run_dt"] = run_dt
print(snap.head(10))
if dry == "N":
    spark.createDataFrame(snap).write.format("delta").mode("append").saveAsTable("fin_cash.ar_aging_snap")

# COMMAND ----------

kc = cm.filter("grp = 'KEY'").select("cust_no").toPandas().cust_no.tolist()
rv = ln[(ln.typ == "UNAPP") & (ln.cust.isin(kc))].groupby("cust").amt_usd.sum().round(2)
rv = rv[rv > 10000]
print("key accts w/ big unapplied:", len(rv))
rv_h = ""
if len(rv) > 0:
    rv_h = "<h4>KEY accounts - unapplied over 10k USD</h4>" + rv.reset_index().to_html(index=False)
print(rv)

# COMMAND ----------

bt = rp.groupby("co").amt.sum().round(2)
lt = ln[ln.typ != "WO"].groupby("co").amt_rc.sum().round(2)
cr = pd.concat([bt, lt], axis=1)
cr.columns = ["bank", "lines"]
cr["diff"] = (cr.bank - cr.lines).round(2)
print(cr)
if (cr["diff"].abs() > 0.01).any():
    print("CO RECON BREAK")

# COMMAND ----------

cs = ln.groupby(["cust", "typ"]).amt_usd.sum().unstack(fill_value=0).round(2)
for c in ["REF", "EXACT", "FIFO", "WO", "UNAPP"]:
    if c not in cs.columns:
        cs[c] = 0.0
cs["applied"] = cs["REF"] + cs["EXACT"] + cs["FIFO"]
cs = cs.sort_values("UNAPP", ascending=False)
print(cs.head(15))

# COMMAND ----------

susp = ln[ln.typ == "SUSP"].merge(rp[["rcpt_id", "payer_nm"]], on="rcpt_id", how="left")
unap = ln[ln.typ == "UNAPP"]
wo = ln[ln.typ == "WO"]
h = "<h3>Cash app run " + run_dt + "</h3>"
h += "<p>receipts: " + str(len(rp)) + " | lines: " + str(len(ln)) + "</p>"
h += "<p>suspense: " + str(len(susp)) + " (" + str(round(susp.amt_usd.sum(), 2)) + " USD)</p>"
h += "<p>unapplied: " + str(len(unap)) + " (" + str(round(unap.amt_usd.sum(), 2)) + " USD)</p>"
h += "<p>write offs: " + str(len(wo)) + " (" + str(round(wo.amt_usd.sum(), 2)) + " USD)</p>"
h += susp[["rcpt_id", "payer_nm", "ccy", "amt_rc"]].to_html(index=False)
h += rv_h
notify_ar(h)

# COMMAND ----------

log_run("cash_app_v3", run_dt, {"rcpt": len(rp), "lines": len(ln), "susp": int(len(susp)), "wo": int(len(wo)), "secs": round(time.time() - t0, 1), "dry": dry})

# COMMAND ----------

# MAGIC %sql
# MAGIC select typ, count(*), round(sum(amt_usd),2) from fin_cash.alloc_lines group by typ order by 1

# COMMAND ----------

# MAGIC %sql
# MAGIC -- select * from fin_cash.alloc_lines where typ = 'WO' order by amt_usd desc
# MAGIC select co, typ, count(*) n from fin_cash.alloc_lines group by co, typ order by co, typ

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"status": "ok", "rcpt": len(rp), "lines": len(ln), "bad": len(bad)}))
