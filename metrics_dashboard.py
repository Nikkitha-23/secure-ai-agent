#!/usr/bin/env python3
"""
Metrics Dashboard
=================
Real-time visualizations of agent performance metrics.
Tracks: latency, decomposition rates, tool routing, synthesis quality.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import statistics

class MetricsDashboard:
    """Interactive metrics dashboard"""
    
    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.summary_file = self.metrics_dir / "summary.json"
        self.metrics_file = self.metrics_dir / "metrics.jsonl"

    def load_summary(self) -> Dict[str, Any]:
        """Load latest metrics summary"""
        if not self.summary_file.exists():
            return {}
        with open(self.summary_file, 'r') as f:
            return json.load(f)

    def load_recent_metrics(self, minutes: int = 60) -> List[Dict]:
        """Load metrics from last N minutes"""
        if not self.metrics_file.exists():
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent = []
        
        with open(self.metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    metric = json.loads(line)
                    try:
                        timestamp = datetime.fromisoformat(metric.get("timestamp", ""))
                        if timestamp > cutoff_time:
                            recent.append(metric)
                    except:
                        pass
        
        return recent

    def generate_html_dashboard(self, output_file: str = "metrics/dashboard.html"):
        """Generate interactive HTML dashboard"""
        
        summary = self.load_summary()
        recent_metrics = self.load_recent_metrics(60)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Metrics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 32px;
        }}
        
        .timestamp {{
            color: #999;
            font-size: 14px;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .card h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .metric:last-child {{
            border-bottom: none;
        }}
        
        .metric-label {{
            color: #666;
            font-weight: 500;
        }}
        
        .metric-value {{
            color: #667eea;
            font-weight: 700;
            font-size: 16px;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            margin-bottom: 10px;
        }}
        
        .charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .status {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 12px;
        }}
        
        .status.good {{
            background: #4caf50;
            color: white;
        }}
        
        .status.warning {{
            background: #ff9800;
            color: white;
        }}
        
        .status.alert {{
            background: #f44336;
            color: white;
        }}
        
        .detail-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        .detail-table th {{
            background: #f5f5f5;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #667eea;
        }}
        
        .detail-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }}
        
        .detail-table tr:hover {{
            background: #f9f9f9;
        }}
        
        footer {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            color: #999;
            font-size: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Agent Metrics Dashboard</h1>
            <p class="timestamp">Last updated: <span id="timestamp">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></p>
        </header>
        
        <!-- KPI Cards -->
        <div class="grid">
            <div class="card">
                <h3>📊 Query Statistics</h3>
                <div class="metric">
                    <span class="metric-label">Total Queries</span>
                    <span class="metric-value">{summary.get('query_statistics', {}).get('total_queries', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Decomposition Rate</span>
                    <span class="metric-value">{summary.get('query_statistics', {}).get('decomposition_rate', 0):.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Decomposed</span>
                    <span class="metric-value">{summary.get('query_statistics', {}).get('decomposed', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Single-Step</span>
                    <span class="metric-value">{summary.get('query_statistics', {}).get('single_step', 0)}</span>
                </div>
            </div>
            
            <div class="card">
                <h3>⏱️ Latency Metrics</h3>
                <div class="metric">
                    <span class="metric-label">Avg Latency</span>
                    <span class="metric-value">{summary.get('latency_metrics', {}).get('avg_latency_ms', 0):.0f}ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Min / Max</span>
                    <span class="metric-value">{summary.get('latency_metrics', {}).get('min_latency_ms', 0):.0f} / {summary.get('latency_metrics', {}).get('max_latency_ms', 0):.0f}ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">P95 Latency</span>
                    <span class="metric-value">{summary.get('latency_metrics', {}).get('p95_latency_ms', 0):.0f}ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">P99 Latency</span>
                    <span class="metric-value">{summary.get('latency_metrics', {}).get('p99_latency_ms', 0):.0f}ms</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🔗 Synthesis Quality</h3>
                <div class="metric">
                    <span class="metric-label">Avg Quality Score</span>
                    <span class="metric-value">{summary.get('synthesis_quality', {}).get('avg_score', 0):.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Synthesis Samples</span>
                    <span class="metric-value">{summary.get('synthesis_quality', {}).get('samples', 0)}</span>
                </div>
            </div>
            
            <div class="card">
                <h3>📍 Routing Confidence</h3>
                <div class="metric">
                    <span class="metric-label">Avg Confidence</span>
                    <span class="metric-value">{summary.get('routing_confidence', {}).get('avg', 0):.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Min / Max</span>
                    <span class="metric-value">{summary.get('routing_confidence', {}).get('min', 0):.3f} / {summary.get('routing_confidence', {}).get('max', 0):.3f}</span>
                </div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="charts">
            <div class="card">
                <h3>📈 Latency Trend</h3>
                <div class="chart-container">
                    <canvas id="latencyChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>🔧 Tool Usage Distribution</h3>
                <div class="chart-container">
                    <canvas id="toolChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>🔀 Query Decomposition</h3>
                <div class="chart-container">
                    <canvas id="decompositionChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Latency Distribution</h3>
                <div class="chart-container">
                    <canvas id="distributionChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Recent Queries -->
        <div class="card">
            <h3>📝 Recent Queries (Last Hour)</h3>
            <table class="detail-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Query</th>
                        <th>Complexity</th>
                        <th>Latency</th>
                        <th>Decomposed</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="recentQueries">
                    <tr><td colspan="6">Loading...</td></tr>
                </tbody>
            </table>
        </div>
        
        <footer>
            <p>🤖 Agent Metrics Dashboard | Real-time monitoring system</p>
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </footer>
    </div>
    
    <script>
        const metrics = {summary};
        const recentMetrics = {json.dumps(recent_metrics)};
        
        // Helper function to create table rows
        function createTableRows(metrics) {{
            return metrics
                .filter(m => m.type !== 'subtask' && m.type !== 'synthesis')
                .slice(-10)
                .map(m => [
                    '<tr>',
                    '<td>' + new Date(m.timestamp).toLocaleTimeString() + '</td>',
                    '<td>' + (m.query || 'N/A') + '</td>',
                    '<td>' + (m.complexity || 'N/A') + '</td>',
                    '<td>' + (m.latency_ms ? m.latency_ms.toFixed(0) + 'ms' : 'N/A') + '</td>',
                    '<td>' + (m.decomposed ? '✅ Yes' : '❌ No') + '</td>',
                    '<td><span class="status good">✓</span></td>',
                    '</tr>'
                ].join(''))
                .join('');
        }}
        
        // Latency Chart
        const latencyCtx = document.getElementById('latencyChart').getContext('2d');
        new Chart(latencyCtx, {{
            type: 'line',
            data: {{
                labels: recentMetrics
                    .filter(m => m.latency_ms !== undefined)
                    .slice(-20)
                    .map(m => new Date(m.timestamp).toLocaleTimeString()),
                datasets: [{{
                    label: 'Latency (ms)',
                    data: recentMetrics
                        .filter(m => m.latency_ms !== undefined)
                        .slice(-20)
                        .map(m => m.latency_ms),
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{ beginAtZero: true }}
                }}
            }}
        }});
        
        // Tool Usage Chart
        const tools = metrics.tool_usage || {{}};
        const toolCtx = document.getElementById('toolChart').getContext('2d');
        new Chart(toolCtx, {{
            type: 'doughnut',
            data: {{
                labels: Object.keys(tools),
                datasets: [{{
                    data: Object.values(tools),
                    backgroundColor: ['#667eea', '#764ba2', '#f093fb', '#4facfe']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ position: 'bottom' }} }}
            }}
        }});
        
        // Decomposition Chart
        const qs = metrics.query_statistics || {{}};
        const decompositionCtx = document.getElementById('decompositionChart').getContext('2d');
        new Chart(decompositionCtx, {{
            type: 'pie',
            data: {{
                labels: ['Decomposed', 'Single-Step'],
                datasets: [{{
                    data: [qs.decomposed || 0, qs.single_step || 0],
                    backgroundColor: ['#4caf50', '#2196f3']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ position: 'bottom' }} }}
            }}
        }});
        
        // Latency Distribution
        const latencies = recentMetrics
            .filter(m => m.latency_ms !== undefined)
            .map(m => m.latency_ms);
        
        const distributionCtx = document.getElementById('distributionChart').getContext('2d');
        new Chart(distributionCtx, {{
            type: 'bar',
            data: {{
                labels: ['<50ms', '50-100ms', '100-200ms', '200-500ms', '>500ms'],
                datasets: [{{
                    label: 'Count',
                    data: [
                        latencies.filter(l => l < 50).length,
                        latencies.filter(l => l >= 50 && l < 100).length,
                        latencies.filter(l => l >= 100 && l < 200).length,
                        latencies.filter(l => l >= 200 && l < 500).length,
                        latencies.filter(l => l >= 500).length
                    ],
                    backgroundColor: '#667eea'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ y: {{ beginAtZero: true }} }}
            }}
        }});
        
        // Recent Queries Table
        const tbody = document.getElementById('recentQueries');
        const tableHTML = createTableRows(recentMetrics);
        tbody.innerHTML = tableHTML || '<tr><td colspan="6" style="text-align: center; color: #999;">No queries in last hour</td></tr>';
    </script>
</body>
</html>"""
        
        Path(output_file).parent.mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Dashboard generated: {output_file}")
        return output_file

    def generate_markdown_report(self, output_file: str = "metrics/report.md"):
        """Generate markdown metrics report"""
        
        summary = self.load_summary()
        recent_metrics = self.load_recent_metrics(60)
        
        report = f"""# Agent Metrics Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total Queries:** {summary.get('query_statistics', {}).get('total_queries', 0)}
- **Decomposition Rate:** {summary.get('query_statistics', {}).get('decomposition_rate', 0):.1f}%
- **Avg Latency:** {summary.get('latency_metrics', {}).get('avg_latency_ms', 0):.0f}ms
- **P95 Latency:** {summary.get('latency_metrics', {}).get('p95_latency_ms', 0):.0f}ms
- **Synthesis Quality:** {summary.get('synthesis_quality', {}).get('avg_score', 0):.3f}/1.0
- **Routing Confidence:** {summary.get('routing_confidence', {}).get('avg', 0):.3f}

## Query Statistics

| Metric | Value |
|--------|-------|
| Total Queries | {summary.get('query_statistics', {}).get('total_queries', 0)} |
| Decomposed Queries | {summary.get('query_statistics', {}).get('decomposed', 0)} |
| Single-Step Queries | {summary.get('query_statistics', {}).get('single_step', 0)} |
| Decomposition Rate | {summary.get('query_statistics', {}).get('decomposition_rate', 0):.1f}% |

## Latency Analysis

| Metric | Value |
|--------|-------|
| Min Latency | {summary.get('latency_metrics', {}).get('min_latency_ms', 0):.0f}ms |
| Avg Latency | {summary.get('latency_metrics', {}).get('avg_latency_ms', 0):.0f}ms |
| P95 Latency | {summary.get('latency_metrics', {}).get('p95_latency_ms', 0):.0f}ms |
| P99 Latency | {summary.get('latency_metrics', {}).get('p99_latency_ms', 0):.0f}ms |
| Max Latency | {summary.get('latency_metrics', {}).get('max_latency_ms', 0):.0f}ms |

## Tool Usage

```
{json.dumps(summary.get('tool_usage', {}), indent=2)}
```

## Routing Confidence

| Metric | Value |
|--------|-------|
| Avg Confidence | {summary.get('routing_confidence', {}).get('avg', 0):.3f} |
| Min Confidence | {summary.get('routing_confidence', {}).get('min', 0):.3f} |
| Max Confidence | {summary.get('routing_confidence', {}).get('max', 0):.3f} |

## Synthesis Quality

| Metric | Value |
|--------|-------|
| Avg Quality Score | {summary.get('synthesis_quality', {}).get('avg_score', 0):.3f} |
| Synthesis Samples | {summary.get('synthesis_quality', {}).get('samples', 0)} |

## Recent Errors

```
{json.dumps(summary.get('error_summary', {}), indent=2)}
```

## Recommendations

1. **Decomposition Rate:** Currently at {summary.get('query_statistics', {}).get('decomposition_rate', 0):.1f}%
   - ✅ Optimal if >20% (complex queries properly decomposed)
   - ⚠️ Check if >50% (may be over-decomposing simple queries)

2. **Latency Performance:** P95 at {summary.get('latency_metrics', {}).get('p95_latency_ms', 0):.0f}ms
   - ✅ Good if <1000ms (under 1 second)
   - ⚠️ Warning if >5000ms (5 seconds, impacts UX)

3. **Synthesis Quality:** Average {summary.get('synthesis_quality', {}).get('avg_score', 0):.3f}
   - ✅ Good if >0.7 (quality synthesis)
   - ⚠️ Review if <0.5 (may need better prompting)

4. **Routing Confidence:** Average {summary.get('routing_confidence', {}).get('avg', 0):.3f}
   - ✅ Good if >0.8 (high confidence routing)
   - ⚠️ Review if <0.7 (uncertain tool selection)

---
Generated by Agent Metrics System
"""
        
        Path(output_file).parent.mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report generated: {output_file}")
        return output_file


if __name__ == "__main__":
    dashboard = MetricsDashboard()
    dashboard.generate_html_dashboard()
    dashboard.generate_markdown_report()
