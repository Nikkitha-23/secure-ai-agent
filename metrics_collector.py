#!/usr/bin/env python3
"""
Metrics Collector & Aggregator
==============================
Collects and stores comprehensive metrics about agent operations.
Records: latency, decomposition, routing, synthesis, quality, memory.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Centralized metrics collection for the agent system"""
    
    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.metrics_file = self.metrics_dir / "metrics.jsonl"
        self.summary_file = self.metrics_dir / "summary.json"
        
        # In-memory aggregation
        self.session_metrics = {
            "total_queries": 0,
            "decomposed_queries": 0,
            "single_step_queries": 0,
            "latencies": [],
            "decomposition_overhead": [],
            "tool_usage": defaultdict(int),
            "routing_confidence": [],
            "synthesis_quality": [],
            "errors": [],
            "timestamps": []
        }

    def record_query(self, query: str, query_type: str, complexity: str, 
                    decomposed: bool, latency: float, num_subtasks: int = 0):
        """Record a single query execution"""
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # First 100 chars
            "query_type": query_type,
            "complexity": complexity,
            "decomposed": decomposed,
            "num_subtasks": num_subtasks,
            "latency_ms": latency * 1000,
        }
        
        # Write to JSONL (append-only log)
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric) + '\n')
        
        # Update in-memory summary
        self.session_metrics["total_queries"] += 1
        self.session_metrics["latencies"].append(latency)
        self.session_metrics["timestamps"].append(datetime.now().isoformat())
        
        if decomposed:
            self.session_metrics["decomposed_queries"] += 1
        else:
            self.session_metrics["single_step_queries"] += 1
        
        logger.info(f"📊 Recorded: {query_type} query ({complexity}) - {latency:.3f}s - decomposed={decomposed}")

    def record_subtask(self, subtask_num: int, task: str, tool: str, 
                      confidence: float, latency: float, success: bool):
        """Record a subtask execution"""
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "type": "subtask",
            "subtask_num": subtask_num,
            "task": task[:100],
            "tool": tool,
            "confidence": confidence,
            "latency_ms": latency * 1000,
            "success": success
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric) + '\n')
        
        # Track tool usage
        self.session_metrics["tool_usage"][tool] += 1
        self.session_metrics["routing_confidence"].append(confidence)
        
        logger.info(f"  ├─ Subtask {subtask_num}: {tool} (confidence={confidence:.2f}) - {latency:.3f}s")

    def record_synthesis(self, quality_score: float, deduplication_effective: bool, 
                        num_sources_combined: int, latency: float):
        """Record synthesis operation"""
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "type": "synthesis",
            "quality_score": quality_score,
            "deduplication_effective": deduplication_effective,
            "sources_combined": num_sources_combined,
            "latency_ms": latency * 1000
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric) + '\n')
        
        self.session_metrics["synthesis_quality"].append(quality_score)
        logger.info(f"  └─ Synthesis: quality={quality_score:.2f}, sources={num_sources_combined}, dedup={deduplication_effective}")

    def record_error(self, error_type: str, error_msg: str, query: str):
        """Record errors for analysis"""
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_type": error_type,
            "error_msg": error_msg[:200],
            "query": query[:100]
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric) + '\n')
        
        self.session_metrics["errors"].append({
            "type": error_type,
            "count": 1
        })
        logger.warning(f"❌ Error: {error_type} - {error_msg}")

    def get_summary(self) -> Dict[str, Any]:
        """Generate summary metrics"""
        
        latencies = self.session_metrics["latencies"]
        confidences = self.session_metrics["routing_confidence"]
        synthesis_scores = self.session_metrics["synthesis_quality"]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "query_statistics": {
                "total_queries": self.session_metrics["total_queries"],
                "decomposed": self.session_metrics["decomposed_queries"],
                "single_step": self.session_metrics["single_step_queries"],
                "decomposition_rate": (
                    self.session_metrics["decomposed_queries"] / 
                    self.session_metrics["total_queries"] * 100
                    if self.session_metrics["total_queries"] > 0 else 0
                )
            },
            "latency_metrics": {
                "avg_latency_ms": (sum(latencies) / len(latencies) * 1000) if latencies else 0,
                "min_latency_ms": (min(latencies) * 1000) if latencies else 0,
                "max_latency_ms": (max(latencies) * 1000) if latencies else 0,
                "p95_latency_ms": self._percentile(latencies, 0.95) * 1000 if latencies else 0,
                "p99_latency_ms": self._percentile(latencies, 0.99) * 1000 if latencies else 0,
            },
            "tool_usage": dict(self.session_metrics["tool_usage"]),
            "routing_confidence": {
                "avg": (sum(confidences) / len(confidences)) if confidences else 0,
                "min": min(confidences) if confidences else 0,
                "max": max(confidences) if confidences else 1,
            },
            "synthesis_quality": {
                "avg_score": (sum(synthesis_scores) / len(synthesis_scores)) if synthesis_scores else 0,
                "samples": len(synthesis_scores)
            },
            "error_summary": self._summarize_errors()
        }
        
        return summary

    def save_summary(self):
        """Save summary to file"""
        summary = self.get_summary()
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"💾 Metrics saved to {self.summary_file}")
        return summary

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * percentile)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    def _summarize_errors(self) -> Dict[str, int]:
        """Summarize errors by type"""
        error_summary = defaultdict(int)
        for error in self.session_metrics["errors"]:
            error_summary[error["type"]] += error["count"]
        return dict(error_summary)

    def print_summary(self):
        """Print formatted summary to console"""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("AGENT METRICS SUMMARY".center(70))
        print("="*70)
        
        # Query statistics
        qs = summary["query_statistics"]
        print(f"\n📊 QUERY STATISTICS:")
        print(f"   Total Queries: {qs['total_queries']}")
        print(f"   Decomposed: {qs['decomposed']} ({qs['decomposition_rate']:.1f}%)")
        print(f"   Single-Step: {qs['single_step']}")
        
        # Latency metrics
        lm = summary["latency_metrics"]
        print(f"\n⏱️ LATENCY METRICS:")
        print(f"   Average: {lm['avg_latency_ms']:.2f}ms")
        print(f"   Min/Max: {lm['min_latency_ms']:.2f}ms / {lm['max_latency_ms']:.2f}ms")
        print(f"   P95 / P99: {lm['p95_latency_ms']:.2f}ms / {lm['p99_latency_ms']:.2f}ms")
        
        # Tool usage
        tu = summary["tool_usage"]
        if tu:
            print(f"\n🔧 TOOL USAGE:")
            for tool, count in sorted(tu.items(), key=lambda x: x[1], reverse=True):
                print(f"   {tool}: {count} times")
        
        # Routing confidence
        rc = summary["routing_confidence"]
        print(f"\n📍 ROUTING CONFIDENCE:")
        print(f"   Average: {rc['avg']:.3f}")
        print(f"   Range: {rc['min']:.3f} - {rc['max']:.3f}")
        
        # Synthesis quality
        sq = summary["synthesis_quality"]
        print(f"\n🔗 SYNTHESIS QUALITY:")
        print(f"   Average Score: {sq['avg_score']:.3f}")
        print(f"   Samples: {sq['samples']}")
        
        # Errors
        if summary["error_summary"]:
            print(f"\n❌ ERRORS:")
            for error_type, count in summary["error_summary"].items():
                print(f"   {error_type}: {count}")
        
        print("\n" + "="*70)


class MetricsAnalyzer:
    """Analyze collected metrics for insights"""
    
    def __init__(self, metrics_file: str = "metrics/metrics.jsonl"):
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> List[Dict]:
        """Load metrics from JSONL file"""
        metrics = []
        if not Path(self.metrics_file).exists():
            return metrics
        
        with open(self.metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        return metrics

    def get_decomposition_stats(self) -> Dict[str, Any]:
        """Analyze decomposition patterns"""
        query_metrics = [m for m in self.metrics if m.get("type") != "subtask" and m.get("type") != "synthesis"]
        
        decomposed = [m for m in query_metrics if m.get("decomposed", False)]
        single_step = [m for m in query_metrics if not m.get("decomposed", True)]
        
        return {
            "total_queries": len(query_metrics),
            "decomposed_count": len(decomposed),
            "decomposition_rate": len(decomposed) / len(query_metrics) * 100 if query_metrics else 0,
            "avg_subtasks": (
                sum(m.get("num_subtasks", 0) for m in decomposed) / len(decomposed)
                if decomposed else 0
            ),
            "complexity_distribution": self._analyze_complexity(query_metrics)
        }

    def get_latency_analysis(self) -> Dict[str, Any]:
        """Analyze latency patterns"""
        query_metrics = [m for m in self.metrics if m.get("type") != "subtask" and m.get("type") != "synthesis"]
        
        if not query_metrics:
            return {}
        
        latencies = [m.get("latency_ms", 0) for m in query_metrics]
        decomposed_latencies = [m.get("latency_ms", 0) for m in query_metrics if m.get("decomposed")]
        single_latencies = [m.get("latency_ms", 0) for m in query_metrics if not m.get("decomposed")]
        
        return {
            "overall": {
                "avg": sum(latencies) / len(latencies) if latencies else 0,
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
            },
            "decomposed": {
                "avg": sum(decomposed_latencies) / len(decomposed_latencies) if decomposed_latencies else 0,
                "count": len(decomposed_latencies)
            },
            "single_step": {
                "avg": sum(single_latencies) / len(single_latencies) if single_latencies else 0,
                "count": len(single_latencies)
            }
        }

    def get_routing_analysis(self) -> Dict[str, Any]:
        """Analyze tool routing patterns"""
        subtask_metrics = [m for m in self.metrics if m.get("type") == "subtask"]
        
        if not subtask_metrics:
            return {}
        
        tool_counts = defaultdict(int)
        tool_confidence = defaultdict(list)
        tool_success = defaultdict(lambda: {"success": 0, "total": 0})
        
        for m in subtask_metrics:
            tool = m.get("tool", "unknown")
            tool_counts[tool] += 1
            tool_confidence[tool].append(m.get("confidence", 0))
            tool_success[tool]["total"] += 1
            if m.get("success", False):
                tool_success[tool]["success"] += 1
        
        return {
            "tool_usage": dict(tool_counts),
            "tool_confidence": {
                tool: {
                    "avg": sum(confs) / len(confs) if confs else 0,
                    "min": min(confs) if confs else 0,
                    "max": max(confs) if confs else 0
                }
                for tool, confs in tool_confidence.items()
            },
            "tool_success_rate": {
                tool: (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
                for tool, stats in tool_success.items()
            }
        }

    def _analyze_complexity(self, metrics: List[Dict]) -> Dict[str, int]:
        """Analyze complexity distribution"""
        complexity_dist = defaultdict(int)
        for m in metrics:
            complexity = m.get("complexity", "unknown")
            complexity_dist[complexity] += 1
        return dict(complexity_dist)

    def print_analysis(self):
        """Print detailed analysis"""
        print("\n" + "="*70)
        print("METRICS ANALYSIS REPORT".center(70))
        print("="*70)
        
        # Decomposition analysis
        decomp = self.get_decomposition_stats()
        print(f"\n🔀 DECOMPOSITION ANALYSIS:")
        print(f"   Total Queries: {decomp['total_queries']}")
        print(f"   Decomposition Rate: {decomp['decomposition_rate']:.1f}%")
        print(f"   Avg Subtasks per Query: {decomp['avg_subtasks']:.1f}")
        print(f"   Complexity Distribution: {decomp['complexity_distribution']}")
        
        # Latency analysis
        latency = self.get_latency_analysis()
        if latency:
            print(f"\n⏱️ LATENCY ANALYSIS:")
            print(f"   Overall Avg: {latency['overall']['avg']:.2f}ms")
            if latency['decomposed']['count'] > 0:
                print(f"   Decomposed Avg: {latency['decomposed']['avg']:.2f}ms ({latency['decomposed']['count']} queries)")
            if latency['single_step']['count'] > 0:
                print(f"   Single-Step Avg: {latency['single_step']['avg']:.2f}ms ({latency['single_step']['count']} queries)")
        
        # Routing analysis
        routing = self.get_routing_analysis()
        if routing:
            print(f"\n📍 ROUTING ANALYSIS:")
            print(f"   Tool Usage: {routing['tool_usage']}")
            print(f"   Tool Confidence:")
            for tool, conf in routing['tool_confidence'].items():
                print(f"      {tool}: avg={conf['avg']:.3f}, range=[{conf['min']:.3f}, {conf['max']:.3f}]")
            print(f"   Success Rates:")
            for tool, rate in routing['tool_success_rate'].items():
                print(f"      {tool}: {rate:.1f}%")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    # Example usage
    collector = MetricsCollector()
    
    # Simulate some metrics
    collector.record_query(
        "Compare CNN and RNN for medical imaging",
        "comparison",
        "complex",
        True,
        0.5,
        num_subtasks=3
    )
    
    collector.record_subtask(1, "Analyze CNN architecture", "pdf", 0.85, 0.15, True)
    collector.record_synthesis(0.92, True, 2, 0.08)
    
    collector.record_query(
        "What is machine learning?",
        "definition",
        "simple",
        False,
        0.1,
        num_subtasks=0
    )
    
    summary = collector.save_summary()
    collector.print_summary()
    
    # Analyze
    analyzer = MetricsAnalyzer()
    analyzer.print_analysis()
