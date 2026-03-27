#!/usr/bin/env python3
"""
Real-Time Agent Monitor
=======================
Monitors agent performance in real-time and provides API endpoint for metrics.
Can be integrated into FastAPI main.py as a dependency injection.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import threading
import time

router = APIRouter(prefix="/metrics", tags=["metrics"])

class RealtimeMonitor:
    """Real-time agent monitoring system"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.query_history = []
        self.subtask_history = []
        self.synthesis_history = []
        self.error_history = []
        self.lock = threading.Lock()
        self.start_time = datetime.now()

    def record_query_start(self, query_id: str, query: str) -> Dict:
        """Record query start time"""
        with self.lock:
            record = {
                "query_id": query_id,
                "query": query[:100],
                "start_time": datetime.now().isoformat(),
                "status": "running"
            }
            self.query_history.append(record)
            if len(self.query_history) > self.max_history:
                self.query_history.pop(0)
            return record

    def record_query_end(self, query_id: str, decomposed: bool, num_subtasks: int, 
                        success: bool, answer_length: int):
        """Record query completion"""
        with self.lock:
            for record in reversed(self.query_history):
                if record["query_id"] == query_id:
                    record["end_time"] = datetime.now().isoformat()
                    record["status"] = "completed" if success else "failed"
                    record["decomposed"] = decomposed
                    record["num_subtasks"] = num_subtasks
                    record["answer_length"] = answer_length
                    
                    # Calculate latency
                    start = datetime.fromisoformat(record["start_time"])
                    end = datetime.fromisoformat(record["end_time"])
                    record["latency_ms"] = (end - start).total_seconds() * 1000
                    break

    def record_subtask(self, query_id: str, subtask_num: int, task: str, 
                      tool: str, confidence: float, latency_ms: float, success: bool):
        """Record subtask execution"""
        with self.lock:
            record = {
                "query_id": query_id,
                "subtask_num": subtask_num,
                "task": task[:100],
                "tool": tool,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "success": success,
                "timestamp": datetime.now().isoformat()
            }
            self.subtask_history.append(record)
            if len(self.subtask_history) > self.max_history:
                self.subtask_history.pop(0)

    def record_synthesis(self, query_id: str, quality_score: float, 
                        num_sources: int, latency_ms: float):
        """Record synthesis operation"""
        with self.lock:
            record = {
                "query_id": query_id,
                "quality_score": quality_score,
                "num_sources": num_sources,
                "latency_ms": latency_ms,
                "timestamp": datetime.now().isoformat()
            }
            self.synthesis_history.append(record)
            if len(self.synthesis_history) > self.max_history:
                self.synthesis_history.pop(0)

    def record_error(self, query_id: str, error_type: str, error_msg: str):
        """Record error"""
        with self.lock:
            record = {
                "query_id": query_id,
                "error_type": error_type,
                "error_msg": error_msg[:200],
                "timestamp": datetime.now().isoformat()
            }
            self.error_history.append(record)
            if len(self.error_history) > self.max_history:
                self.error_history.pop(0)

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        with self.lock:
            recent_queries = self.query_history[-100:]
            recent_subtasks = self.subtask_history[-200:]
            recent_synthesis = self.synthesis_history[-100:]
            
            # Calculate statistics
            completed_queries = [q for q in recent_queries if q.get("status") == "completed"]
            failed_queries = [q for q in recent_queries if q.get("status") == "failed"]
            decomposed_queries = [q for q in recent_queries if q.get("decomposed", False)]
            
            latencies = [q.get("latency_ms", 0) for q in completed_queries if "latency_ms" in q]
            confidences = [s.get("confidence", 0) for s in recent_subtasks if "confidence" in s]
            quality_scores = [s.get("quality_score", 0) for s in recent_synthesis if "quality_score" in s]
            
            # Calculate aggregates
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # Tool distribution
            tool_usage = {}
            for subtask in recent_subtasks:
                tool = subtask.get("tool", "unknown")
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
            
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "query_statistics": {
                    "total_queries": len(self.query_history),
                    "completed": len(completed_queries),
                    "failed": len(failed_queries),
                    "success_rate": len(completed_queries) / len(self.query_history) * 100 if self.query_history else 0,
                    "decomposed": len(decomposed_queries),
                    "decomposition_rate": len(decomposed_queries) / len(self.query_history) * 100 if self.query_history else 0
                },
                "latency_metrics": {
                    "avg_ms": avg_latency,
                    "min_ms": min(latencies) if latencies else 0,
                    "max_ms": max(latencies) if latencies else 0,
                    "p95_ms": self._percentile(latencies, 0.95) if latencies else 0,
                    "p99_ms": self._percentile(latencies, 0.99) if latencies else 0
                },
                "tool_usage": tool_usage,
                "routing_confidence": {
                    "avg": avg_confidence,
                    "min": min(confidences) if confidences else 0,
                    "max": max(confidences) if confidences else 1
                },
                "synthesis_quality": {
                    "avg_score": avg_quality,
                    "samples": len(quality_scores)
                },
                "errors": {
                    "total": len(self.error_history),
                    "recent": len([e for e in self.error_history if (
                        datetime.now() - datetime.fromisoformat(e["timestamp"])
                    ).total_seconds() < 300])  # Last 5 minutes
                }
            }

    def get_recent_queries(self, limit: int = 20) -> List[Dict]:
        """Get recent queries"""
        with self.lock:
            return self.query_history[-limit:]

    def get_query_detail(self, query_id: str) -> Dict[str, Any]:
        """Get detailed stats for a specific query"""
        with self.lock:
            query_record = None
            for record in reversed(self.query_history):
                if record["query_id"] == query_id:
                    query_record = record
                    break
            
            if not query_record:
                return {}
            
            subtasks = [s for s in self.subtask_history if s["query_id"] == query_id]
            synthesis = [s for s in self.synthesis_history if s["query_id"] == query_id]
            errors = [e for e in self.error_history if e["query_id"] == query_id]
            
            return {
                "query": query_record,
                "subtasks": subtasks,
                "synthesis": synthesis,
                "errors": errors
            }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * percentile)
        return sorted_data[min(idx, len(sorted_data) - 1)]


# Global monitor instance
monitor = RealtimeMonitor()


# ── API Endpoints ──────────────────────────────────────────────────────────

@router.get("/stats")
async def get_metrics_stats():
    """Get current metrics statistics"""
    return monitor.get_current_stats()


@router.get("/queries/recent")
async def get_recent_queries(limit: int = 20):
    """Get recent queries"""
    return {
        "queries": monitor.get_recent_queries(limit),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/queries/{query_id}")
async def get_query_detail(query_id: str):
    """Get detailed stats for a specific query"""
    detail = monitor.get_query_detail(query_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Query not found")
    return detail


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = monitor.get_current_stats()
    return {
        "status": "healthy",
        "uptime_seconds": stats["uptime_seconds"],
        "total_queries": stats["query_statistics"]["total_queries"],
        "success_rate": stats["query_statistics"]["success_rate"]
    }


@router.get("/dashboard")
async def get_dashboard_data():
    """Get all data needed for dashboard"""
    return {
        "current_stats": monitor.get_current_stats(),
        "recent_queries": monitor.get_recent_queries(50),
        "timestamp": datetime.now().isoformat()
    }


# Helper functions for integration

def record_agent_query(query_id: str, query: str):
    """Record start of agent query"""
    return monitor.record_query_start(query_id, query)


def record_agent_completion(query_id: str, decomposed: bool, num_subtasks: int,
                           success: bool, answer_length: int):
    """Record completion of agent query"""
    monitor.record_query_end(query_id, decomposed, num_subtasks, success, answer_length)


def record_agent_subtask(query_id: str, subtask_num: int, task: str, tool: str,
                        confidence: float, latency_ms: float, success: bool):
    """Record subtask execution"""
    monitor.record_subtask(query_id, subtask_num, task, tool, confidence, latency_ms, success)


def record_agent_synthesis(query_id: str, quality_score: float, num_sources: int, latency_ms: float):
    """Record synthesis"""
    monitor.record_synthesis(query_id, quality_score, num_sources, latency_ms)


def record_agent_error(query_id: str, error_type: str, error_msg: str):
    """Record error"""
    monitor.record_error(query_id, error_type, error_msg)


if __name__ == "__main__":
    # Test the monitor
    import uuid
    
    query_id = str(uuid.uuid4())
    monitor.record_query_start(query_id, "Compare CNN and RNN")
    
    monitor.record_subtask(query_id, 1, "Analyze CNN architecture", "pdf", 0.85, 150.5, True)
    monitor.record_subtask(query_id, 2, "Analyze RNN architecture", "pdf", 0.82, 140.2, True)
    
    monitor.record_synthesis(query_id, 0.92, 2, 50.3)
    monitor.record_query_end(query_id, True, 2, True, 512)
    
    print("\n📊 MONITOR STATISTICS\n")
    print(json.dumps(monitor.get_current_stats(), indent=2))
    
    print("\n📝 QUERY DETAIL\n")
    print(json.dumps(monitor.get_query_detail(query_id), indent=2))
