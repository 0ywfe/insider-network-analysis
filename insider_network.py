## insider trading network analysis using SEC Form 4 filings
## replicating arXiv:2512.18918v1

import requests
import pandas as pd
import numpy as np
import networkx as nx
import json, time, os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

START_YEAR = 2014
END_YEAR = 2024
WINDOW_WEEKS = 1
MIN_TRADES = 5
TOP_N_CENTRAL = 50
ODDBALL_THRESHOLD = 2.0

OUTPUT_DIR = Path("results")
CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "form4_cache.json"
RESULTS_FILE = OUTPUT_DIR / "network_results.json"
CLUSTERS_FILE = OUTPUT_DIR / "suspicious_clusters.csv"
STATS_FILE = OUTPUT_DIR / "network_stats.json"

EDGAR_HEADERS = {
    "User-Agent": "Academic Research Network Analysis",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}


def fetch_form4_filings(start_year, end_year, use_cache=True):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if use_cache and CACHE_FILE.exists():
        print("Loading Form 4 data from cache...")
        df = pd.read_json(CACHE_FILE)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        print(f"  Loaded {len(df):,} transactions from cache")
        return df

    print(f"Fetching Form 4 filings from EDGAR ({start_year}-{end_year})...")
    print("This will take 30-60 minutes for the full dataset.\n")

    all_filings = []
    total_fetched = 0

    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):

            current_quarter = (datetime.now().year - start_year) * 4 + (datetime.now().month - 1) // 3
            this_quarter = (year - start_year) * 4 + quarter - 1
            if this_quarter > current_quarter:
                continue

            print(f"  Fetching {year} Q{quarter}...")
            quarter_filings = fetch_quarter_form4(year, quarter)
            all_filings.extend(quarter_filings)
            total_fetched += len(quarter_filings)
            print(f"    Got {len(quarter_filings):,} transactions (total: {total_fetched:,})")

            time.sleep(0.5)

    if not all_filings:
        print("No filings fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_filings)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df = df.dropna(subset=['transaction_date', 'filer_cik'])
    df = df.sort_values('transaction_date')

    df.to_json(CACHE_FILE, orient='records', date_format='iso')
    print(f"\nCached {len(df):,} transactions to {CACHE_FILE}")

    return df


def fetch_quarter_form4(year, quarter):

    quarter_starts = {1: f"{year}-01-01", 2: f"{year}-04-01", 3: f"{year}-07-01", 4: f"{year}-10-01"}
    quarter_ends = {1: f"{year}-03-31", 2: f"{year}-06-30", 3: f"{year}-09-30", 4: f"{year}-12-31"}

    start_date = quarter_starts[quarter]
    end_date = quarter_ends[quarter]

    filings = []
    from_idx = 0
    batch = 100
    total = None

    while True:
        url = (
            "https://efts.sec.gov/LATEST/search-index?"
            f"forms=4&dateRange=custom&startdt={start_date}&enddt={end_date}"
            f"&from={from_idx}&size={batch}"
        )

        try:
            resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"    EDGAR error at offset {from_idx}: {exc}")
            break

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        if total is None:
            total = data.get("hits", {}).get("total", {}).get("value", 0)

        for hit in hits:
            src = hit.get("_source", {})
            filing_date = src.get("period_of_report") or src.get("file_date", "")
            entity = src.get("entity_name", "Unknown")
            if isinstance(entity, list):
                entity = entity[0] if entity else "Unknown"
            accession = hit.get("_id", "")
            cik = accession.split("-")[0].lstrip("0") if accession else ""

            display_names = src.get("display_names", [])
            filer_name = display_names[0] if display_names else entity

            if filing_date and len(filing_date) >= 10:
                filings.append({
                    "filer_name": str(filer_name),
                    "filer_cik": cik,
                    "issuer_name": str(entity),
                    "transaction_date": filing_date[:10],
                    "accession": accession,
                })

        from_idx += batch
        if total and from_idx >= total:
            break
        if from_idx >= 10000:
            break

        time.sleep(0.15)

    return filings


def build_insider_network(df, window_weeks=1):
    ## builds weighted graph where edge weight = number of weeks both insiders filed in

    print(f"\nBuilding insider network...")
    print(f"  Transactions: {len(df):,}")
    print(f"  Unique filers: {df['filer_cik'].nunique():,}")
    print(f"  Date range: {df['transaction_date'].min().date()} to {df['transaction_date'].max().date()}")

    trade_counts = df['filer_cik'].value_counts()
    active_filers = trade_counts[trade_counts >= MIN_TRADES].index
    df_filtered = df[df['filer_cik'].isin(active_filers)].copy()
    print(f"  Filers with >= {MIN_TRADES} trades: {len(active_filers):,}")

    df_filtered['week'] = df_filtered['transaction_date'].dt.to_period('W')

    print("  Computing co-filing pairs by week...")
    edge_weights = defaultdict(int)

    weeks = df_filtered['week'].unique()
    total_weeks = len(weeks)

    for i, week in enumerate(weeks):

        week_filers = df_filtered[df_filtered['week'] == week]['filer_cik'].unique()

        if len(week_filers) < 2:
            continue

        for j in range(len(week_filers)):
            for k in range(j + 1, len(week_filers)):
                pair = tuple(sorted([week_filers[j], week_filers[k]]))
                edge_weights[pair] += 1

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{total_weeks} weeks...")

    print(f"  Building graph from {len(edge_weights):,} potential edges...")
    G = nx.Graph()

    for cik in active_filers:
        filer_data = df_filtered[df_filtered['filer_cik'] == cik].iloc[0]
        G.add_node(cik,
                   name=filer_data['filer_name'],
                   trade_count=int(trade_counts[cik]))

    ## only keep edges where insiders co-filed in 10+ weeks
    min_weight = 10
    for (cik1, cik2), weight in edge_weights.items():
        if weight >= min_weight:
            if G.has_node(cik1) and G.has_node(cik2):
                G.add_edge(cik1, cik2, weight=weight)

    print(f"\nNetwork built:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Paper target: 4,650 nodes, 7,007 edges")

    return G


def compute_centrality(G):

    print("\nComputing centrality measures...")

    degree_cent = nx.degree_centrality(G)

    strength = {node: sum(d['weight'] for _, _, d in G.edges(node, data=True))
                for node in G.nodes()}

    print("  Betweenness centrality (approximate)...")
    if G.number_of_nodes() > 1000:
        betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), weight='weight')
    else:
        betweenness = nx.betweenness_centrality(G, weight='weight')

    print("  PageRank...")
    pagerank = nx.pagerank(G, weight='weight', max_iter=200)

    print("  Clustering coefficients...")
    clustering = nx.clustering(G, weight='weight')

    rows = []
    for node in G.nodes():
        rows.append({
            "cik": node,
            "name": G.nodes[node].get('name', 'Unknown'),
            "trade_count": G.nodes[node].get('trade_count', 0),
            "degree": G.degree(node),
            "degree_cent": degree_cent[node],
            "strength": strength[node],
            "betweenness": betweenness[node],
            "pagerank": pagerank[node],
            "clustering": clustering[node],
        })

    df_cent = pd.DataFrame(rows).sort_values('pagerank', ascending=False)
    print(f"  Top node by PageRank: {df_cent.iloc[0]['name']} (CIK: {df_cent.iloc[0]['cik']})")

    return df_cent


def oddball_detection(G):
    ## power law deviation in ego-net edge density

    print("\nRunning OddBall anomaly detection...")

    rows = []
    for node in G.nodes():

        neighbors = list(G.neighbors(node))
        degree = len(neighbors)
        if degree < 2:
            continue

        egonet = G.subgraph([node] + neighbors)
        egonet_edges = egonet.number_of_edges()
        total_weight = sum(G[node][nb].get('weight', 1) for nb in neighbors)

        rows.append({
            "cik": node,
            "name": G.nodes[node].get('name', 'Unknown'),
            "degree": degree,
            "egonet_edges": egonet_edges,
            "total_weight": total_weight,
        })

    if not rows:
        return pd.DataFrame()

    df_odd = pd.DataFrame(rows)

    ## fit power law: log(egonet_edges) ~ a * log(degree) + b
    log_degree = np.log1p(df_odd['degree'])
    log_edges = np.log1p(df_odd['egonet_edges'])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_degree, log_edges)
    expected_log_edges = slope * log_degree + intercept
    residuals = log_edges - expected_log_edges

    std_resid = residuals.std()
    df_odd['oddball_score'] = np.abs(residuals) / (std_resid + 1e-10)
    df_odd['residual'] = residuals

    anomalous = df_odd[df_odd['oddball_score'] > ODDBALL_THRESHOLD].sort_values('oddball_score', ascending=False)
    print(f"  Power law fit: slope={slope:.3f}, R²={r_value**2:.3f}")
    print(f"  Anomalous nodes (score > {ODDBALL_THRESHOLD}): {len(anomalous):,}")

    return df_odd.sort_values('oddball_score', ascending=False)


def null_model_validation(G, n_shuffles=10):

    print(f"\nValidating against null model ({n_shuffles} shuffles)...")

    real_edges = G.number_of_edges()
    components = nx.connected_components(G)
    real_largest = max(len(c) for c in components) if G.number_of_nodes() > 0 else 0

    null_edges = []
    null_largest = []

    for i in range(n_shuffles):

        G_null = G.copy(); edges = list(G_null.edges()); np.random.shuffle(edges); G_null = nx.Graph(); G_null.add_edges_from(edges)
        null_edges.append(G_null.number_of_edges())
        components_null = nx.connected_components(G_null)
        null_largest.append(max(len(c) for c in components_null) if G_null.number_of_nodes() > 0 else 0)

        if (i + 1) % 5 == 0:
            print(f"  Shuffle {i + 1}/{n_shuffles}...")

    null_edges_mean = np.mean(null_edges)
    null_edges_std = np.std(null_edges)
    null_largest_mean = np.mean(null_largest)
    null_largest_std = np.std(null_largest)

    z_edges = (real_edges - null_edges_mean) / (null_edges_std + 1e-10)
    z_largest = (real_largest - null_largest_mean) / (null_largest_std + 1e-10)

    print(f"\n  Real edges: {real_edges:,} | Null mean: {null_edges_mean:.1f} | Z-score: {z_edges:.1f}")
    print(f"  Real largest component: {real_largest:,} | Null mean: {null_largest_mean:.1f} | Z-score: {z_largest:.1f}")
    print(f"  Paper target: Z-scores >1000")

    return {
        "real_edges": real_edges,
        "null_edges_mean": null_edges_mean,
        "null_edges_std": null_edges_std,
        "z_score_edges": z_edges,
        "real_largest_comp": real_largest,
        "null_largest_mean": null_largest_mean,
        "z_score_largest": z_largest,
    }


def find_suspicious_clusters(G, centrality_df, oddball_df):

    print("\nIdentifying suspicious clusters...")

    components = list(nx.connected_components(G))
    components_sorted = sorted(components, key=len, reverse=True)

    rows = []
    for i, component in enumerate(components_sorted[:20]):

        subgraph = G.subgraph(component)
        density = nx.density(subgraph)
        avg_weight = np.mean([d['weight'] for _, _, d in subgraph.edges(data=True)]) if subgraph.edges() else 0
        members = [G.nodes[n].get('name', n) for n in component]

        member_pagerank = centrality_df[centrality_df['cik'].isin(component)]['pagerank'].mean() if len(centrality_df) > 0 else 0
        member_oddball = oddball_df[oddball_df['cik'].isin(component)]['oddball_score'].max() if len(oddball_df) > 0 else 0

        rows.append({
            "cluster_id": i + 1,
            "size": len(component),
            "edges": subgraph.number_of_edges(),
            "density": round(density, 4),
            "avg_edge_weight": round(avg_weight, 2),
            "max_oddball": round(member_oddball, 2),
            "avg_pagerank": round(member_pagerank, 6),
            "members_sample": ", ".join(members[:5]) + ("..." if len(members) > 5 else ""),
        })

    df_clusters = pd.DataFrame(rows)

    ## flag suspicious: high density + high oddball
    df_clusters['suspicious'] = (
        (df_clusters['density'] > df_clusters['density'].quantile(0.75)) &
        (df_clusters['max_oddball'] > ODDBALL_THRESHOLD)
    )

    print(f"  Total clusters: {len(df_clusters)}")
    print(f"  Suspicious clusters: {df_clusters['suspicious'].sum()}")

    return df_clusters


def main():

    print("Insider trading network analysis")
    print(f"Replicating arXiv:2512.18918v1\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = fetch_form4_filings(START_YEAR, END_YEAR, use_cache=True)
    if df.empty:
        print("No data.")
        return

    print(f"\nDataset: {len(df):,} transactions, {df['filer_cik'].nunique():,} filers")
    print(f"Date range: {df['transaction_date'].min().date()} to {df['transaction_date'].max().date()}")

    G = build_insider_network(df, WINDOW_WEEKS)

    centrality_df = compute_centrality(G)
    print(f"\nTop 10 insiders by PageRank:")
    print(centrality_df[['name', 'degree', 'strength', 'pagerank']].head(10).to_string(index=False))

    oddball_df = oddball_detection(G)
    if not oddball_df.empty:
        print(f"\nTop 10 anomalous nodes:")
        print(oddball_df[['name', 'degree', 'egonet_edges', 'oddball_score']].head(10).to_string(index=False))

    null_results = null_model_validation(G, n_shuffles=10)

    clusters_df = find_suspicious_clusters(G, centrality_df, oddball_df)
    clusters_df.to_csv(CLUSTERS_FILE, index=False)

    results = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "start_year": START_YEAR,
            "end_year": END_YEAR,
            "total_transactions": len(df),
            "unique_filers": int(df['filer_cik'].nunique()),
            "paper": "arXiv:2512.18918v1",
        },
        "network": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "components": nx.number_connected_components(G),
            "paper_nodes": 4650,
            "paper_edges": 7007,
        },
        "null_model": null_results,
        "top_central_nodes": centrality_df.head(TOP_N_CENTRAL).to_dict(orient='records'),
        "top_anomalous_nodes": oddball_df.head(50).to_dict(orient='records') if not oddball_df.empty else [],
        "suspicious_clusters": clusters_df[clusters_df['suspicious']].to_dict(orient='records'),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Clusters saved to {CLUSTERS_FILE}")
    print(f"\nNetwork: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"Paper target: 4,650 nodes, 7,007 edges")
    print(f"Suspicious clusters: {clusters_df['suspicious'].sum()}")


if __name__ == "__main__":
    main()
