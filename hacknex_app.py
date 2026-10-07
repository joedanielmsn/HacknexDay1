"""
Streamlit UI for the log attack detector.
Run with:   pip install streamlit pandas scikit-learn
            streamlit run hacknex_app.py
"""
import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Log Attack Detector", page_icon="🛡️", layout="wide")

REASONS = {
    "rule_failed":   "Too many failed logins",
    "rule_bytes":    "Too much data downloaded at once",
    "rule_slow":     "Too much data over an hour (slow theft)",
    "rule_accounts": "One IP used many different accounts",
    "rule_takeover": "Login SUCCEEDED after many failures (possible takeover)",
}
RULE_COLS = list(REASONS.keys())


def human_size(n):
    for unit in ["bytes", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:,.1f} {unit}"
        n /= 1024


@st.cache_data(show_spinner=False)
def load_logs(source):
    df = pd.read_csv(source, parse_dates=["time"])
    df["row_id"] = range(len(df))  # remember original row order (for the answer key)
    return df.sort_values("time").reset_index(drop=True)


def detect(df, window, max_failed, max_bytes_window, max_bytes_hour,
           max_accounts, service_accounts, contamination):
    df = df.copy()
    df["window"] = df["time"].dt.floor(window)

    work = df[~df["user"].isin(service_accounts)].copy()
    work["is_failed"] = work["result"] == "failed"
    work["is_ok_login"] = (work["action"] == "login") & (work["result"] == "success")

    counts = work.groupby(["ip", "window"]).agg(
        failed_logins=("is_failed", "sum"),
        ok_logins=("is_ok_login", "sum"),
        accounts=("user", "nunique"),
        bytes_out=("bytes", "sum"),
    ).reset_index()

    dl = work[work["action"] == "download"].set_index("time")
    if len(dl):
        hb = dl.groupby("ip")["bytes"].rolling("60min").sum().rename("bytes_hour").reset_index()
        hb["window"] = hb["time"].dt.floor(window)
        hb = hb.groupby(["ip", "window"])["bytes_hour"].max().reset_index()
        counts = counts.merge(hb, on=["ip", "window"], how="left")
    else:
        counts["bytes_hour"] = 0
    counts["bytes_hour"] = counts["bytes_hour"].fillna(0)

    counts["rule_failed"] = counts["failed_logins"] > max_failed
    counts["rule_bytes"] = counts["bytes_out"] > max_bytes_window
    counts["rule_slow"] = (counts["bytes_hour"] > max_bytes_hour) & ~counts["rule_bytes"]
    counts["rule_accounts"] = counts["accounts"] >= max_accounts
    counts["rule_takeover"] = (counts["failed_logins"] > max_failed) & (counts["ok_logins"] > 0)

    features = ["failed_logins", "ok_logins", "accounts", "bytes_out", "bytes_hour"]
    model = IsolationForest(contamination=contamination, random_state=0)
    counts["ml_alert"] = model.fit_predict(counts[features]) == -1

    alerts = counts[counts[RULE_COLS].any(axis=1) | counts["ml_alert"]].sort_values(["ip", "window"]).copy()

    if alerts.empty:
        return df, counts, alerts, pd.DataFrame()

    gap = alerts.groupby("ip")["window"].diff() > pd.Timedelta("15min")
    alerts["incident"] = (gap | alerts["ip"].ne(alerts["ip"].shift())).cumsum()
    incidents = alerts.groupby("incident").agg(
        ip=("ip", "first"), start=("window", "min"), end=("window", "max"),
        failed_logins=("failed_logins", "sum"), accounts=("accounts", "max"),
        bytes_out=("bytes_out", "sum"), bytes_hour=("bytes_hour", "max"),
        ml_alert=("ml_alert", "any"), **{c: (c, "any") for c in RULE_COLS},
    )
    incidents["end"] = incidents["end"] + pd.Timedelta(window)
    incidents["n_rules"] = incidents[RULE_COLS].sum(axis=1)

    def conf(r):
        if r["n_rules"] >= 2:
            return "HIGH"
        if r["n_rules"] == 1 and r["ml_alert"]:
            return "MEDIUM"
        return "LOW"

    incidents["confidence"] = incidents.apply(conf, axis=1)
    incidents["why"] = incidents.apply(
        lambda r: "; ".join(
            [t for c, t in REASONS.items() if r[c]]
            + (["Unusual pattern (ML)" if r["n_rules"] else "Unusual pattern (ML only, no rule fired)"]
               if r["ml_alert"] else [])
        ), axis=1)
    incidents = incidents.sort_values(["n_rules", "bytes_out"], ascending=False)
    return df, counts, alerts, incidents


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛡️ Settings")

st.sidebar.subheader("Data")
up_logs = st.sidebar.file_uploader("Log file (CSV)", type="csv")
up_labels = st.sidebar.file_uploader("Answer key (optional CSV)", type="csv")
default_log = "logins_10k.csv"
default_labels = "logins_10k_labels.csv"

st.sidebar.subheader("Rules")
window = st.sidebar.selectbox("Time window", ["5min", "10min", "15min", "30min"], index=1)
max_failed = st.sidebar.slider("Max failed logins / window", 5, 500, 50)
max_bytes_window = st.sidebar.slider("Max MB / window", 10, 2000, 500, step=10) * 1_000_000
max_bytes_hour = st.sidebar.slider("Max MB / rolling hour", 10, 2000, 400, step=10) * 1_000_000
max_accounts = st.sidebar.slider("Accounts per IP (spraying)", 2, 30, 5)

st.sidebar.subheader("Machine learning")
contamination = st.sidebar.slider("Expected anomaly share", 0.005, 0.20, 0.02, step=0.005)

# ------------------------------------------------------------------ load data
st.title("Log Attack Detector")
st.caption("Rules + Isolation Forest, grouped into per-IP incidents.")

if up_logs is not None:
    df_raw = load_logs(up_logs)
elif os.path.exists(default_log):
    df_raw = load_logs(default_log)
else:
    st.info("Upload a log CSV in the sidebar (columns: time, user, ip, action, result, bytes).")
    st.stop()

all_users = sorted(df_raw["user"].unique())
default_svc = [u for u in ["backup_svc", "updates_svc"] if u in all_users]
service_accounts = st.sidebar.multiselect("Service accounts (ignored)", all_users, default=default_svc)

df, counts, alerts, incidents = detect(
    df_raw, window, max_failed, max_bytes_window, max_bytes_hour,
    max_accounts, set(service_accounts), contamination)

# ------------------------------------------------------------------ KPIs
flagged_keys = alerts[["ip", "window"]].assign(flagged=True) if len(alerts) else pd.DataFrame(columns=["ip", "window", "flagged"])
df = df.merge(flagged_keys, on=["ip", "window"], how="left")
df["flagged"] = df["flagged"].fillna(False).astype(bool)
logins = df[df["action"] == "login"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Log rows", f"{len(df):,}")
c2.metric("Incidents", len(incidents))
c3.metric("Flagged windows", len(alerts))
c4.metric("Logins flagged", f"{int(logins['flagged'].sum()):,}")
c5.metric("Logins not flagged", f"{len(logins) - int(logins['flagged'].sum()):,}")

tab_inc, tab_time, tab_ip, tab_score = st.tabs(
    ["🚨 Incidents", "📈 Timeline", "🔎 IP drill-down", "✅ Answer key"])

# ------------------------------------------------------------------ incidents
with tab_inc:
    if incidents.empty:
        st.success("No incidents found with the current settings.")
    else:
        levels = st.multiselect("Confidence", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
        view = incidents[incidents["confidence"].isin(levels)]

        table = pd.DataFrame({
            "Confidence": view["confidence"],
            "IP": view["ip"],
            "Start": view["start"],
            "End": view["end"],
            "Failed logins": view["failed_logins"].astype(int),
            "Accounts": view["accounts"].astype(int),
            "Downloaded": view["bytes_out"].map(human_size),
            "Peak 1h": view["bytes_hour"].map(human_size),
            "Why flagged": view["why"],
        })

        def colour(v):
            return {"HIGH": "background-color:#f8d7da", "MEDIUM": "background-color:#fff3cd",
                    "LOW": "background-color:#e2e3e5"}.get(v, "")

        st.dataframe(table.style.map(colour, subset=["Confidence"]),
                     use_container_width=True, hide_index=True)
        st.download_button("Download incidents (CSV)", table.to_csv(index=False),
                           "incidents.csv", "text/csv")

# ------------------------------------------------------------------ timeline
with tab_time:
    per_window = counts.groupby("window").agg(
        failed_logins=("failed_logins", "sum"), bytes_out=("bytes_out", "sum"))
    st.subheader("Failed logins per window")
    st.line_chart(per_window["failed_logins"])
    st.subheader("Data downloaded per window (MB)")
    st.bar_chart(per_window["bytes_out"] / 1e6)
    if not incidents.empty:
        st.subheader("Incidents by confidence")
        st.bar_chart(incidents["confidence"].value_counts())

# ------------------------------------------------------------------ IP drill-down
with tab_ip:
    ip_choices = list(incidents["ip"].unique()) if not incidents.empty else sorted(df["ip"].unique())
    ip = st.selectbox("IP address", ip_choices)
    sub = df[df["ip"] == ip]
    a, b, c = st.columns(3)
    a.metric("Events", len(sub))
    b.metric("Failed", int((sub["result"] == "failed").sum()))
    c.metric("Downloaded", human_size(sub.loc[sub["action"] == "download", "bytes"].sum()))
    st.write("Activity over time")
    st.bar_chart(sub.groupby("window").size())
    st.write("Raw events")
    st.dataframe(sub.drop(columns=["row_id", "window"]).head(1000),
                 use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ answer key
with tab_score:
    labels_src = up_labels if up_labels is not None else (default_labels if os.path.exists(default_labels) else None)
    if labels_src is None:
        st.info("Upload an answer-key CSV (with a `label` column) in the sidebar to score the detector.")
    else:
        labels = pd.read_csv(labels_src)["label"]
        if len(labels) != len(df_raw):
            st.warning("Answer key length doesn't match the log file.")
        else:
            df["label"] = df["row_id"].map(lambda i: labels.iloc[i])
            summary = df.groupby("label")["flagged"].agg(rows="size", flagged="sum")
            summary["% flagged"] = (100 * summary["flagged"] / summary["rows"]).round(1)
            st.dataframe(summary, use_container_width=True)
            st.bar_chart(summary["% flagged"])
