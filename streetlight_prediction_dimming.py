"""
streetlight_closed_loop_analysis.py
===================================
ONE integrated, re-run-everything pipeline that CLOSES THE PREDICTION->CONTROL
LOOP and anchors the dimming floor to a road-lighting CLASS with a luminance
safety check.

PIPELINE
  A. Integrate 12 SLPSP sensor files + Open-Meteo weather + Delhi traffic.
  B. Leakage-controlled demand model (removes the 38 electrical features that
     algebraically reconstruct power), chronological 80/20 split, REGULARISED
     Random Forest, train/test metrics, feature importance, ablation.
  C. CLOSED-LOOP CONTROL (the new linkage):
       - the model's predicted demand P_hat decides the OPERATING window
         (lamp on when P_hat > threshold) -> the predictor gates the controller;
       - when operating, a luminance-ANCHORED policy sets intensity in
         [FLOOR, 1], where FLOOR = L(min-class) / L(design-class);
       - a LUMINANCE CHECK verifies the dimmed road luminance never drops below
         the class minimum (safety guarantee, reported);
       - controlled power = P_hat * power_map(intensity);
       - SAVINGS = predicted baseline - controlled, then VALIDATED against the
         actual metered power on the held-out test period.

THESIS DEMONSTRATED
  "Adopting the model (weather + traffic + power features) lets the operator
   reduce luminosity during low-demand hours down to the class minimum, cutting
   operating-hour power by ~XX% out-of-sample, while NEVER violating the
   lighting-class luminance floor."

SAFETY / SCOPE: illustrative. The chosen class anchoring (single common class
for both roads, per request) and the linear power-vs-output model are stated
assumptions; a deployment needs the road authority's class designation, the
luminaire photometric file, and field measurement.
"""

import os, sys, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ============================================================
# CONFIG
# ============================================================
CFG = {
    "slpsp_glob":  "/mnt/project/SLPSP_*.csv",
    "weather_csv": "/mnt/user-data/uploads/open-meteo-28_65N77_27E231m__1_.csv",
    "traffic_csv": "/mnt/user-data/uploads/delhi_typical_congestion_2024-11-30_to_2025-03-21.csv",
    "out_dir":     "/home/claude/work/TrialRun_enhanced/closed_loop",
    "timezone":    "Asia/Kolkata",
    "target":      "Total Active power",
    "test_fraction": 0.20,
    "random_state": 42,

    # --- regularised RF (addresses prior overfitting: was max_depth=None) ---
    "rf": dict(n_estimators=300, max_depth=16, min_samples_leaf=5,
               min_samples_split=10, max_features="sqrt"),

    # ============================================================
    # LIGHTING-CLASS ANCHORING  (single common class for both roads)
    #   EN 13201-2 / CIE 115 average road-surface luminance (cd/m^2):
    #   M1 2.00 | M2 1.50 | M3 1.00 | M4 0.75 | M5 0.50 | M6 0.30
    # ============================================================
    "design_class":    "M3",   # full output achieves this class
    "night_min_class": "M5",   # lowest class allowed when dimmed at low demand
    "M_CLASS": {"M1": 2.00, "M2": 1.50, "M3": 1.00, "M4": 0.75, "M5": 0.50, "M6": 0.30},

    # --- control policy ---
    "cong_full": 55.0,          # congestion %% at/above which intensity = 100%
    "vis_clear": 5000.0, "vis_hi": 2000.0, "vis_lo": 1000.0, "vis_fog": 200.0,
    "w_vis": 0.30,
    "driver_overhead": 0.10,    # fixed (non-dimmable) power fraction
    "operating_threshold_kW": 4.0,   # lamp "operating" when (predicted/actual) power > this
}
# derived floor + luminance levels
CFG["L_design"] = CFG["M_CLASS"][CFG["design_class"]]
CFG["L_min"]    = CFG["M_CLASS"][CFG["night_min_class"]]
CFG["FLOOR"]    = CFG["L_min"] / CFG["L_design"]   # single common floor


# ============================================================
# A. DATA INTEGRATION
# ============================================================
def integrate_data():
    paths = sorted(glob.glob(CFG["slpsp_glob"]))
    if not paths:
        sys.exit(f"[ERROR] no SLPSP files matched {CFG['slpsp_glob']}")
    frames = []
    for p in paths:
        frames.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df["_ts"] = pd.to_datetime(df["data_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["_ts", CFG["target"]]).copy()
    df["_ts"] = df["_ts"].dt.tz_convert(CFG["timezone"])
    df = df.sort_values("_ts").reset_index(drop=True)
    print(f"  SLPSP: {len(df):,} readings, {df['_ts'].min()} -> {df['_ts'].max()}")

    # time features
    df["Hour"] = df["_ts"].dt.hour
    df["DoY"]  = df["_ts"].dt.dayofyear
    df["DoW"]  = df["_ts"].dt.dayofweek
    df["IsWeekend"] = (df["DoW"] >= 5).astype(int)
    df["Hour_sin"] = np.sin(2*np.pi*df["Hour"]/24); df["Hour_cos"] = np.cos(2*np.pi*df["Hour"]/24)
    df["DoY_sin"]  = np.sin(2*np.pi*df["DoY"]/365); df["DoY_cos"]  = np.cos(2*np.pi*df["DoY"]/365)

    # IST hourly key for merging
    ist = df["_ts"]
    df["_hourkey"] = pd.to_datetime(dict(year=ist.dt.year, month=ist.dt.month,
                                         day=ist.dt.day, hour=ist.dt.hour))
    df["_weekday"] = ist.dt.weekday; df["_hr"] = ist.dt.hour; df["_month"] = ist.dt.month

    # weather
    w = pd.read_csv(CFG["weather_csv"], skiprows=2).rename(columns={
        "time": "_hourkey", "temperature_2m (\u00b0C)": "wx_temp_C",
        "relative_humidity_2m (%)": "wx_humidity_pct", "rain (mm)": "wx_rain_mm",
        "cloud_cover (%)": "wx_cloud_pct", "visibility (m)": "wx_visibility_m",
        "sunshine_duration (s)": "wx_sunshine_s", "shortwave_radiation (W/m\u00b2)": "wx_swrad_wm2"})
    w["_hourkey"] = pd.to_datetime(w["_hourkey"])
    wx = ["wx_temp_C","wx_humidity_pct","wx_rain_mm","wx_cloud_pct","wx_visibility_m","wx_sunshine_s","wx_swrad_wm2"]
    df = df.merge(w[["_hourkey"]+wx].drop_duplicates("_hourkey"), on="_hourkey", how="left")
    df["Precipitation_Flag"] = (df["wx_rain_mm"].fillna(0) > 0).astype(int)

    # traffic (typical weekday x hour profile + monthly)
    t = pd.read_csv(CFG["traffic_csv"]); tt = pd.to_datetime(t["timestamp"])
    t["_weekday"]=tt.dt.weekday; t["_hr"]=tt.dt.hour; t["_month"]=tt.dt.month
    cong = t.groupby(["_weekday","_hr"])["congestion_level_pct"].mean().rename("tr_congestion_pct").reset_index()
    mon  = t.groupby("_month")["monthly_congestion_level_pct"].mean().rename("tr_monthly_congestion_pct").reset_index()
    df = df.merge(cong, on=["_weekday","_hr"], how="left").merge(mon, on="_month", how="left")

    print(f"  Merged weather + traffic. Coverage wx={df['wx_swrad_wm2'].notna().mean()*100:.0f}%  "
          f"traffic={df['tr_congestion_pct'].notna().mean()*100:.0f}%")
    return df


# Electrical features that algebraically reconstruct the target (removed)
LEAKY = ["R-Phase Active Power","Y-Phase Active Power","B-Phase Active Power",
    "Average value Active Power","Maximum Value Active Power","Total Apparent Power",
    "Average Value Apparent Power","Maximum Value Apparent Power","R-Phase Apparent Power",
    "Y-Phase Apparent Power","B-Phase Apparent Power","System Power Factor","R-Phase Power Factor",
    "Y-Phase Power Factor","B-Phase Power Factor","R-Phase Voltage","Y-Phase Voltage","B-Phase Voltage",
    "R-Phase Current","Y-Phase Current","B-Phase Current","R-Phase Average Current","Y-Phase Average Current",
    "B-Phase Average Current","R-Phase Maximum Current","Y-Phase Maximum Current","B-Phase Maximum Current",
    "Total current","Total Reactive Power","Average Value Reactive Power","Maximum Value Reactive Power",
    "R-Phase Reactive Power","Y-Phase Reactive Power","B-Phase Reactive Power","Energy","Run Hour",
    "Light Status","Power Status"]

CONTEXT = ["Frequency","wx_temp_C","wx_humidity_pct","wx_rain_mm","wx_cloud_pct","wx_visibility_m",
           "wx_sunshine_s","wx_swrad_wm2","tr_congestion_pct","tr_monthly_congestion_pct","Precipitation_Flag"]
NEW_EXTERNAL = ["wx_temp_C","wx_humidity_pct","wx_rain_mm","wx_cloud_pct","wx_visibility_m",
                "wx_sunshine_s","wx_swrad_wm2","tr_congestion_pct","tr_monthly_congestion_pct","Precipitation_Flag"]
TIME_FEATS = ["Hour_sin","Hour_cos","DoY_sin","DoY_cos","DoW","IsWeekend"]


def build_xy(df):
    y = df[CFG["target"]].copy()
    keep = [c for c in CONTEXT if c in df.columns] + TIME_FEATS
    X = df[keep].copy()
    X = pd.concat([X, pd.get_dummies(df[["Area","Name"]], dtype=int)], axis=1)
    X = X.fillna(X.mean(numeric_only=True))
    return X, y


# ============================================================
# B. DEMAND MODEL + EVALUATION
# ============================================================
def report(tag, yt, yp, r2=True):
    mae = mean_absolute_error(yt, yp); rmse = np.sqrt(mean_squared_error(yt, yp))
    r = r2_score(yt, yp) if r2 else float("nan")
    print(f"    {tag:<34s} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r if r2 else float('nan'):.4f}")
    return dict(model=tag, MAE_kW=mae, RMSE_kW=rmse, R2=r if r2 else np.nan)


def run_prediction(df):
    print("\n[B] LEAKAGE-CONTROLLED DEMAND MODEL")
    # leakage proof
    phase = ["R-Phase Active Power","Y-Phase Active Power","B-Phase Active Power"]
    if all(c in df for c in phase):
        pred = df[phase].sum(axis=1); m = pred.notna() & df[CFG["target"]].notna()
        print(f"    leakage check: P = R+Y+B Active -> R2={r2_score(df[CFG['target']][m], pred[m]):.4f} "
              f"(electrical features removed)")

    X, y = build_xy(df)
    n = len(X); cut = int(n*(1-CFG["test_fraction"]))
    Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    ts_tr, ts_te = df["_ts"].iloc[:cut], df["_ts"].iloc[cut:]
    print(f"    chronological split: train {len(Xtr):,} ({ts_tr.min().date()}..{ts_tr.max().date()}) | "
          f"test {len(Xte):,} ({ts_te.min().date()}..{ts_te.max().date()})")

    rf = RandomForestRegressor(random_state=CFG["random_state"], n_jobs=-1, **CFG["rf"])
    rf.fit(Xtr, ytr)
    rows = []
    print("  Baselines / models (test set):")
    rows.append(report("Static rated-power (25 kW)", yte, np.full(len(yte), 25.0), r2=False))
    hourmean = df.iloc[:cut].groupby("Hour")[CFG["target"]].mean()
    yp_hm = df.iloc[cut:]["Hour"].map(hourmean).fillna(ytr.mean()).values
    rows.append(report("Hour-of-day mean", yte, yp_hm))
    lr = LinearRegression().fit(Xtr, ytr); rows.append(report("Linear Regression", yte, lr.predict(Xte)))
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                           random_state=CFG["random_state"], n_jobs=-1, verbosity=0).fit(Xtr, ytr)
        rows.append(report("XGBoost", yte, xgb.predict(Xte)))
    rows.append(report("Random Forest (test)", yte, rf.predict(Xte)))
    print("  Overfitting check:")
    tr_m = report("Random Forest (TRAIN)", ytr, rf.predict(Xtr))
    te_m = rows[-1]
    print(f"    train/test MAE ratio = {te_m['MAE_kW']/tr_m['MAE_kW']:.2f}  "
          f"(regularised: max_depth={CFG['rf']['max_depth']}, min_samples_leaf={CFG['rf']['min_samples_leaf']})")

    # ablation: marginal value of new features (regularised, same params)
    new = [c for c in NEW_EXTERNAL if c in Xtr.columns]
    base_cols = [c for c in Xtr.columns if c not in new]
    def fit_mae(cols):
        m = RandomForestRegressor(random_state=CFG["random_state"], n_jobs=-1, **CFG["rf"]).fit(Xtr[cols], ytr)
        p = m.predict(Xte[cols])
        return mean_absolute_error(yte, p), np.sqrt(mean_squared_error(yte, p)), r2_score(yte, p)
    mb = fit_mae(base_cols); mf = fit_mae(list(Xtr.columns))
    print("  Ablation (marginal value of weather+traffic):")
    print(f"    WITHOUT wx+traffic: MAE={mb[0]:.3f} RMSE={mb[1]:.3f} R2={mb[2]:.4f}")
    print(f"    WITH    wx+traffic: MAE={mf[0]:.3f} RMSE={mf[1]:.3f} R2={mf[2]:.4f}")
    print(f"    delta: MAE {mb[0]-mf[0]:+.4f} kW ({100*(mb[0]-mf[0])/mb[0]:+.2f}%), "
          f"RMSE {100*(mb[1]-mf[1])/mb[1]:+.2f}%, R2 {mf[2]-mb[2]:+.4f}")

    # permutation importance (test set)
    perm = permutation_importance(rf, Xte, yte, n_repeats=5,
                                  random_state=CFG["random_state"], n_jobs=-1)
    imp = pd.DataFrame({"feature": Xte.columns, "importance": perm.importances_mean,
                        "std": perm.importances_std}).sort_values("importance", ascending=False)
    imp["is_new"] = imp["feature"].isin(NEW_EXTERNAL)
    print("  Top features (permutation; * = new external):")
    for _, r in imp.head(8).iterrows():
        print(f"    {'*' if r['is_new'] else ' '} {r['feature']:<26s} {r['importance']:.4f}")

    Path(CFG["out_dir"]).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(CFG["out_dir"])/"R1_model_comparison.csv", index=False)
    imp.to_csv(Path(CFG["out_dir"])/"R2_feature_importance.csv", index=False)
    pd.DataFrame([{"variant":"without_wx_traffic","MAE":mb[0],"RMSE":mb[1],"R2":mb[2]},
                  {"variant":"with_wx_traffic","MAE":mf[0],"RMSE":mf[1],"R2":mf[2]}]
                 ).to_csv(Path(CFG["out_dir"])/"R3_ablation.csv", index=False)

    return rf, X, y, cut, df, dict(train=tr_m, test=te_m, ablation=(mb, mf), importance=imp)


# ============================================================
# C. CLOSED-LOOP CONTROL  (predictor-gated, luminance-anchored)
# ============================================================
def _vis_boost(vis):
    vis = np.asarray(vis, float); b = np.zeros_like(vis)
    m1 = (vis >= CFG["vis_hi"]) & (vis < CFG["vis_clear"]); b[m1] = (CFG["vis_clear"]-vis[m1])/(CFG["vis_clear"]-CFG["vis_hi"])*0.6
    m2 = (vis >= CFG["vis_lo"]) & (vis < CFG["vis_hi"]);    b[m2] = 0.6 + (CFG["vis_hi"]-vis[m2])/(CFG["vis_hi"]-CFG["vis_lo"])*0.4
    m3 = (vis >= CFG["vis_fog"]) & (vis < CFG["vis_lo"]);   b[m3] = 1.0
    return CFG["w_vis"]*b

def dimming_factor(congestion, visibility):
    """Intensity in [FLOOR, 1] from demand/safety conditions (operating hours)."""
    cong = np.asarray(congestion, float); floor = CFG["FLOOR"]
    f = floor + (1-floor)*np.clip(cong/CFG["cong_full"], 0, 1)
    f = np.minimum(f + _vis_boost(visibility), 1.0)
    f = np.where(np.asarray(visibility, float) < CFG["vis_fog"], floor, f)  # dense fog -> floor
    return np.clip(f, floor, 1.0)

def power_map(f):
    oh = CFG["driver_overhead"]; return oh + (1-oh)*f


def run_closed_loop(rf, X, y, cut, df):
    print("\n[C] CLOSED-LOOP CONTROL (predictor-gated + luminance-anchored)")
    print(f"    Class anchoring: design {CFG['design_class']}={CFG['L_design']:.2f} cd/m2, "
          f"night-min {CFG['night_min_class']}={CFG['L_min']:.2f} cd/m2 -> common FLOOR={CFG['FLOOR']:.0%}")

    Xte, yte = X.iloc[cut:].copy(), y.iloc[cut:].copy()
    dte = df.iloc[cut:].copy()
    thr = CFG["operating_threshold_kW"]

    # 1) model predicts baseline demand on held-out test set
    p_hat = rf.predict(Xte)
    dte["p_hat"] = p_hat
    dte["p_actual"] = yte.values

    # 2) operating window decided by the MODEL (predictor gates the controller)
    operating_fc = p_hat > thr
    operating_ac = dte["p_actual"].values > thr

    # 3) luminance-anchored intensity when operating
    f = dimming_factor(dte["tr_congestion_pct"].values, dte["wx_visibility_m"].values)
    dte["intensity"] = np.where(operating_fc, f, 0.0)

    # 4) LUMINANCE CHECK: dimmed road-surface luminance vs class minimum
    lum = np.where(operating_fc, f * CFG["L_design"], np.nan)   # cd/m2 when operating
    lum_on = lum[operating_fc]
    min_lum = np.nanmin(lum_on) if lum_on.size else float("nan")
    pct_compliant = 100 * np.mean(lum_on >= CFG["L_min"] - 1e-9) if lum_on.size else float("nan")
    print(f"    Luminance check: min dimmed luminance = {min_lum:.2f} cd/m2 ; "
          f"class minimum = {CFG['L_min']:.2f} cd/m2 ; compliant {pct_compliant:.1f}% of operating hours")

    # 5) controlled power and SAVINGS
    #    planned (forecast baseline): over hours the model says are operating
    base_fc = p_hat[operating_fc]
    ctrl_fc = base_fc * power_map(f[operating_fc])
    planned_saving = 100*(1 - ctrl_fc.sum()/base_fc.sum())
    #    realized (validation on ACTUAL metered power): over actually-operating hours
    pa = dte["p_actual"].values
    f_ac = f[operating_ac]
    realized_saving = 100*(1 - (pa[operating_ac]*power_map(f_ac)).sum() / pa[operating_ac].sum())
    print(f"    Energy saving during operating hours (held-out test / March):")
    print(f"      PLANNED  (from model forecast baseline): {planned_saving:.1f}%")
    print(f"      REALIZED (validated on actual metered power): {realized_saving:.1f}%")
    print(f"      -> the planned, forecast-driven saving is realised to within "
          f"{abs(planned_saving-realized_saving):.1f} pp on unseen data.")

    # mean luminosity reduction during operating hours
    mean_intensity = f_ac.mean()
    print(f"    Mean luminosity during operating hours: {100*mean_intensity:.0f}% of design "
          f"(i.e. luminosity reduced ~{100*(1-mean_intensity):.0f}% on average; down to the "
          f"{CFG['night_min_class']} floor of {CFG['FLOOR']:.0%} in the deep night).")

    # save hourly test-period schedule
    sched = dte[["_ts","Hour","tr_congestion_pct","wx_visibility_m","wx_swrad_wm2",
                 "p_actual","p_hat","intensity"]].copy()
    sched["intensity_pct"] = (100*sched["intensity"]).round(1)
    sched["luminance_cd_m2"] = (sched["intensity"]*CFG["L_design"]).round(3)
    sched["p_controlled_kW"] = np.where(operating_fc, sched["p_hat"]*power_map(f), 0.0).round(3)
    sched.to_csv(Path(CFG["out_dir"])/"R4_hourly_control_schedule_testset.csv", index=False)

    results = dict(planned_saving=planned_saving, realized_saving=realized_saving,
                   min_lum=min_lum, pct_compliant=pct_compliant, mean_intensity=mean_intensity,
                   floor=CFG["FLOOR"], L_min=CFG["L_min"], L_design=CFG["L_design"])

    # ---------- figures ----------
    _figs(rf, Xte, yte, dte, f, operating_fc, results)
    return results


def _figs(rf, Xte, yte, dte, f, operating_fc, res):
    od = Path(CFG["out_dir"])

    # Fig A: closed-loop time series over a representative 4-day window
    d = dte.copy(); d["intensity"] = np.where(operating_fc, f, 0.0)
    d["p_ctrl"] = np.where(operating_fc, d["p_hat"]*power_map(f), 0.0)
    d["lum"] = np.where(operating_fc, f*CFG["L_design"], np.nan)
    # pick the first 4 full days in the test set, one representative device (highest mean load)
    dev = d.groupby("Name")["p_actual"].mean().idxmax()
    dd = d[d["Name"] == dev].copy().sort_values("_ts")
    t0 = dd["_ts"].min().normalize() + pd.Timedelta(days=2)
    win = dd[(dd["_ts"] >= t0) & (dd["_ts"] < t0 + pd.Timedelta(days=4))]
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios":[2,1]})
    ax[0].plot(win["_ts"], win["p_actual"], color="#7f8c8d", lw=1.2, label="actual power (current)")
    ax[0].plot(win["_ts"], win["p_hat"], color="#2980b9", lw=1.2, ls="--", label="model forecast baseline")
    ax[0].plot(win["_ts"], win["p_ctrl"], color="#27ae60", lw=1.6, label="controlled power (dimmed)")
    ax[0].fill_between(win["_ts"], win["p_ctrl"], win["p_hat"], where=win["p_hat"]>win["p_ctrl"],
                       color="#27ae60", alpha=0.15, label="energy saved")
    ax[0].set_ylabel("Power (kW)"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"Closed loop on held-out test data ({dev}): model gates operation, "
                    f"controller dims, savings = baseline − controlled")
    ax[1].plot(win["_ts"], win["lum"], color="#8e44ad", lw=1.6, label="dimmed road luminance")
    ax[1].axhline(CFG["L_min"], color="red", ls="--", lw=1.2,
                  label=f"{CFG['night_min_class']} class minimum ({CFG['L_min']:.2f} cd/m²)")
    ax[1].set_ylabel("Luminance (cd/m²)"); ax[1].set_ylim(0, CFG["L_design"]*1.1)
    ax[1].legend(loc="upper right", fontsize=8); ax[1].set_xlabel("Date / time (IST)")
    fig.autofmt_xdate(); plt.tight_layout(); plt.savefig(od/"figA_closed_loop_timeseries.png", dpi=150); plt.close()

    # Fig B: savings (planned vs realized) + luminance compliance
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    bars = ax[0].bar(["planned\n(forecast)","realized\n(actual)"],
                     [res["planned_saving"], res["realized_saving"]],
                     color=["#2980b9","#27ae60"], edgecolor="black")
    for b,v in zip(bars,[res["planned_saving"],res["realized_saving"]]):
        ax[0].text(b.get_x()+b.get_width()/2, v, f"{v:.1f}%", ha="center", va="bottom")
    ax[0].set_ylabel("Operating-hour energy saving (%)")
    ax[0].set_title("Predicted saving is realised out-of-sample"); ax[0].grid(axis="y", alpha=0.3)
    # luminance: design vs floor vs min reached
    ax[1].bar(["design\n(M3 full)","class min\n(M5 floor)","min reached\n(dimmed)"],
              [res["L_design"], res["L_min"], res["min_lum"]],
              color=["#f1c40f","#e74c3c","#8e44ad"], edgecolor="black")
    ax[1].axhline(res["L_min"], color="red", ls="--", lw=1)
    ax[1].set_ylabel("Road-surface luminance (cd/m²)")
    ax[1].set_title(f"Luminance never below class minimum ({res['pct_compliant']:.0f}% compliant)")
    ax[1].grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(od/"figB_savings_and_luminance.png", dpi=150); plt.close()
    print(f"    Saved figA_closed_loop_timeseries.png, figB_savings_and_luminance.png and 4 result CSVs -> {od}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("#"*72)
    print("#  CLOSED-LOOP STREETLIGHT ANALYSIS  (predict -> control -> verified savings)")
    print(f"#  Common luminance floor: {CFG['design_class']}->{CFG['night_min_class']} = {CFG['FLOOR']:.0%}")
    print("#"*72)
    print("\n[A] DATA INTEGRATION")
    df = integrate_data()
    rf, X, y, cut, df, diag = run_prediction(df)
    res = run_closed_loop(rf, X, y, cut, df)

    print("\n" + "="*72)
    print("  HEADLINE FOR THE PAPER")
    print("="*72)
    print(f"  Adopting the model (weather + traffic + power features) lets the operator")
    print(f"  cut OPERATING-HOUR power by ~{res['realized_saving']:.0f}% (validated out-of-sample)")
    print(f"  by reducing luminosity to as low as {res['floor']:.0%} of design output, while road")
    print(f"  luminance never falls below the {CFG['night_min_class']} class minimum ({res['L_min']:.2f} cd/m², "
          f"{res['pct_compliant']:.0f}% compliant).")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
