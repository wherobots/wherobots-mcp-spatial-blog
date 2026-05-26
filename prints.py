"""
prints.py — Print/summary helpers for CIV MCP Demo Scenarios.

Keeps the main notebook focused on queries and visualizations.
Each helper accepts Spark DataFrames and uses Spark APIs (count, first, agg)
so callers do not need to materialize a pandas DataFrame just to print stats.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# ── Scenario 1 ───────────────────────────────────────────────────────────────
def print_s1(df_s1: DataFrame) -> None:
    print(f"Multi-hazard assets returned: {df_s1.count()}")


# ── Scenario 2 ───────────────────────────────────────────────────────────────
def print_s2(df_s2: DataFrame) -> None:
    print(f"Top social-media hotspot assets: {df_s2.count()}")


# ── Scenario 3 ───────────────────────────────────────────────────────────────
def print_s3(df_s3: DataFrame) -> None:
    print(f"Active geofences: {df_s3.count()}")


# ── Scenario 4 ───────────────────────────────────────────────────────────────
def print_s4(df_s4: DataFrame) -> None:
    total = df_s4.count()
    inside = df_s4.filter(F.col("geofence_status").startswith("Inside")).count()
    print(f"Highest-concern assets (multi-hazard + social alert): {total}")
    print(f"   - Inside a monitoring geofence : {inside}")
    print(f"   - Outside geofences            : {total - inside}")


# ── Scenario 5 ───────────────────────────────────────────────────────────────
def print_s5(df_s5: DataFrame) -> None:
    agg = df_s5.agg(
        F.sum("asset_count").alias("total"),
        F.sum("flood_exposed").alias("flood"),
        F.sum("wildfire_exposed").alias("wildfire"),
        F.sum("multi_hazard").alias("multi_hazard"),
        F.sum("social_alert").alias("social_alert"),
    ).first()
    print(f"Total Tier-5 assets    : {agg['total']:,}")
    print(f"Flood-exposed          : {agg['flood']:,}")
    print(f"Wildfire-exposed       : {agg['wildfire']:,}")
    print(f"Multi-hazard           : {agg['multi_hazard']:,}")
    print(f"Social alert triggered : {agg['social_alert']:,}")


# ── Scenario 6 ───────────────────────────────────────────────────────────────
def print_s6(df_s6: DataFrame, df_s6_summary: DataFrame) -> None:
    print(f"Assets with full Mar-Sep window (7 months): {df_s6.count()}")
    print("\nHazard Window Distribution:")
    df_s6_summary.show(truncate=False)


# ── Scenario 7 ───────────────────────────────────────────────────────────────
def print_s7(df_s7: DataFrame) -> None:
    print(f"Assets near hospitals/schools with social signal: {df_s7.count()}")


# ── Scenario 8 ───────────────────────────────────────────────────────────────
def print_s8(df_s8_summary: DataFrame, df_s8_top: DataFrame) -> None:
    print("Communication Tower Summary:")
    df_s8_summary.show(truncate=False)
    print("\nTop 20 towers in wildfire zones by tweet activity:")
    df_s8_top.drop("geometry").show(truncate=False)


# ── Scenario 9 ───────────────────────────────────────────────────────────────
def print_s9(sedona, df_s9: DataFrame) -> None:
    total_dams = sedona.sql(
        "SELECT COUNT(*) AS n FROM wherobots.geoint.enhanced_vulnerability_civ "
        "WHERE class = 'dam'"
    ).first()["n"]
    flood_dams = sedona.sql(
        "SELECT COUNT(*) AS n FROM wherobots.geoint.enhanced_vulnerability_civ "
        "WHERE class = 'dam' AND flood_count > 0"
    ).first()["n"]
    social_dams = df_s9.filter(F.col("social_alert_flag") == True).count()  # noqa: E712
    pct = (100 * flood_dams / total_dams) if total_dams else 0.0
    print(f" Total dams in dataset    : {total_dams:,}")
    print(f"Dams in flood zones      : {flood_dams:,} ({pct:.1f}%)")
    print(f"Social-alert dams        : {social_dams}")
    print("\nTop 20 by Compound Risk Score:")


# ── Scenario 10 ──────────────────────────────────────────────────────────────
def print_s10(has_bc: bool) -> None:
    if not has_bc:
        print("Border Corridor table not available — showing CIV profile only")
    print("Cross-Demo Profile Comparison:")


# ── Scenario D2 ──────────────────────────────────────────────────────────────
def print_d2(df_thresholds: DataFrame, df_d2: DataFrame) -> None:
    row = df_thresholds.first()
    avg_flood = float(row["avg_flood_threshold"])
    avg_wf = float(row["avg_wildfire_threshold"])
    top = df_d2.first()
    print("Dynamic thresholds used:")
    print(f"   Avg flood_count    = {avg_flood:.2f}")
    print(f"   Avg wildfire_count = {avg_wf:.2f}")
    print(
        f"\nClass with highest compound exposure: "
        f"{top['class'].upper()} ({int(top['compound_hazard_assets']):,} assets)"
    )


# ── Scenario D3 ──────────────────────────────────────────────────────────────
def print_d3(df_d3: DataFrame) -> None:
    print(f"High-criticality assets with >50% negative tweet ratio: {df_d3.count()}")
    print("\nThreat Monitoring Interpretation:")
    print("   Assets with BOTH high criticality (Tier 4-5) AND dominant negative sentiment")
    print("   represent the intersection of strategic importance and public concern —")
    print("   indicating potential active reporting of failures, near-miss events, or")
    print("   widespread awareness of vulnerability that adversaries could exploit.")


# ── Scenario D4 ──────────────────────────────────────────────────────────────
def print_d4(df_d4: DataFrame, by_class: DataFrame) -> None:
    total = df_d4.count()
    social = df_d4.filter(F.col("social_alert_flag") == True).count()  # noqa: E712
    pct = (100 * social / total) if total else 0.0
    print(f"Tier-5 assets with Mar-Sep peak window  : {total:,}")
    print(f"Of those with social alert flag set     : {social:,} ({pct:.1f}%)")
    print("\nBreakdown by class:")
    by_class.show(truncate=False)


# ── Scenario D5 ──────────────────────────────────────────────────────────────
def print_d5(df_d5: DataFrame, top_fence_id: str) -> None:
    print("Geofence Hotspot Rankings:")
    df_d5.select(
        "geofence_id",
        "high_tier_assets",
        "fence_negative_tweets",
        "avg_sentiment",
        "risk_score",
    ).show(truncate=False)
    top = df_d5.first()
    print(f"\nHIGHEST-RISK GEOFENCE: {top_fence_id}")
    print(f"   Composite risk score : {int(top['risk_score']):,}")
    print(f"   Total assets inside  : {int(top['total_assets']):,}")
    print(f"   Tier 4-5 assets      : {int(top['high_tier_assets']):,}")
    print(f"   Fence negative tweets: {int(top['fence_negative_tweets']):,}")
    print(f"   Avg sentiment        : {float(top['avg_sentiment']):.4f}")
    print("\nAsset class breakdown inside top geofence:")
