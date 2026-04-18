# Insider Trading Network Analysis

Graph-based anomaly detection on 2.9M SEC Form 4 filings to find suspicious insider trading clusters. Replicating arXiv:2512.18918v1.

## How it works

Downloads Form 4 insider transaction filings from SEC EDGAR (2014-2024), builds a weighted network where insiders are connected if they filed in the same weeks, then runs centrality analysis and OddBall anomaly detection to find clusters with unusual connectivity patterns.

The paper's network had 4,650 nodes and 7,007 edges with null model Z-scores above 1000.

## How to run

```bash
pip install -r requirements.txt
python3 insider_network.py
```

First run takes 30-60 minutes to fetch data from EDGAR. Gets cached locally after that.

## Output

- `results/network_results.json` - Network stats, top central nodes, anomalous nodes, suspicious clusters
- `results/suspicious_clusters.csv` - Ranked clusters with density and OddBall scores

## What it does

1. Fetches Form 4 filings from SEC EDGAR quarterly (2014-2024)
2. Builds a weighted graph based on weekly co-filing (edge weight = number of weeks both insiders filed)
3. Computes centrality (degree, betweenness, PageRank, clustering coefficient)
4. Runs OddBall anomaly detection - fits a power law to ego-net edge density and flags nodes that deviate
5. Validates against a shuffled null model
6. Identifies and ranks suspicious clusters by density + anomaly score

## Data source

SEC EDGAR (free, public, no API key). Rate-limited to 10 requests/second.

## License

MIT
