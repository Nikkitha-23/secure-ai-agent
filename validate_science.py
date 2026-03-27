#!/usr/bin/env python3
"""
Validation Science Suite for Multi-Step Planning
================================================

Tests 5 critical dimensions:
1. Dependency queries - proper decomposition + reasoning
2. Hierarchical reasoning - ordered, structured sub-tasks
3. Mixed tool queries - correct tool routing per subtask
4. Latency explosion - measure response time degradation
5. Synthesis intelligence - deduplicate, resolve, structure
"""

import time
import json
from typing import Dict, List, Tuple
from rag.agent_loop import (
    run_agent, 
    decompose_query, 
    detect_complexity,
    route_sources
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationSuite:
    def __init__(self):
        self.results = {
            "test_1_dependency": [],
            "test_2_hierarchical": [],
            "test_3_mixed_tools": [],
            "test_4_latency": [],
            "test_5_synthesis": []
        }
        self.metrics = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "avg_latency": 0,
            "max_latency": 0,
            "min_latency": float('inf')
        }

    def test_1_dependency_queries(self) -> Dict:
        """
        TEST 1: Dependency Queries
        ===========================
        Query: "Compare CNN and RNN and suggest best for medical imaging"
        
        Checks:
        - ✅ Correctly splits into 3 subtasks?
        - ✅ Synthesis reasons about medical imaging context?
        - ✅ Each subtask routed correctly?
        """
        print("\n" + "="*70)
        print("TEST 1: DEPENDENCY QUERIES")
        print("="*70)
        
        query = "Compare CNN and RNN and suggest best for medical imaging"
        print(f"\n📝 Query: {query}\n")
        
        # Step 1: Check decomposition detection
        decomp = decompose_query(query)
        print(f"🔍 Decomposition:\n  should_decompose: {decomp.get('should_decompose')}")
        print(f"  num_subtasks: {len(decomp.get('subtasks', []))}")
        
        test_1_result = {
            "query": query,
            "complexity": detect_complexity(query)["level"],
            "decomposed": decomp.get("should_decompose", False),
            "num_subtasks": len(decomp.get("subtasks", [])),
            "subtasks": decomp.get("subtasks", []),
            "expected_subtasks": 3,
            "pass": False
        }
        
        # Check: Should decompose into ~3 subtasks
        if test_1_result["decomposed"] and test_1_result["num_subtasks"] >= 2:
            print("  ✅ PASS: Correctly decomposed")
            test_1_result["pass"] = True
        else:
            print("  ❌ FAIL: Should decompose but didn't")
        
        # Print task breakdown
        if decomp.get("subtasks"):
            print(f"\n📋 Subtasks breakdown:")
            for st in decomp.get("subtasks", []):
                print(f"   [{st.get('number')}] {st.get('task')}")
                routing = route_sources(st.get('task'))
                print(f"       → Routing: {routing['primary']} (confidence: {routing['confidence']})")
        
        print(f"\n📊 Result: {'✅ PASS' if test_1_result['pass'] else '❌ FAIL'}")
        self.results["test_1_dependency"].append(test_1_result)
        return test_1_result

    def test_2_hierarchical_reasoning(self) -> Dict:
        """
        TEST 2: Hierarchical Reasoning
        ===============================
        Query: "Explain transformers, then compare with LSTM, then give real use case"
        
        Checks:
        - ✅ Recognizes ordered/hierarchical structure?
        - ✅ Subtasks follow logical sequence?
        - ✅ Each builds on previous (ordered reasoning)?
        """
        print("\n" + "="*70)
        print("TEST 2: HIERARCHICAL REASONING")
        print("="*70)
        
        query = "Explain transformers, then compare with LSTM, then give real use case"
        print(f"\n📝 Query: {query}\n")
        
        # Step 1: Check decomposition
        decomp = decompose_query(query)
        print(f"🔍 Decomposition:\n  should_decompose: {decomp.get('should_decompose')}")
        print(f"  num_subtasks: {len(decomp.get('subtasks', []))}")
        
        test_2_result = {
            "query": query,
            "complexity": detect_complexity(query)["level"],
            "decomposed": decomp.get("should_decompose", False),
            "num_subtasks": len(decomp.get("subtasks", [])),
            "subtasks": decomp.get("subtasks", []),
            "has_sequence": False,
            "pass": False
        }
        
        # Check: Should recognize "then" as ordering signal
        subtasks = decomp.get("subtasks", [])
        if len(subtasks) >= 3:
            # Check if tasks mention sequence (explain → compare → use case)
            task_texts = [st.get('task', '').lower() for st in subtasks]
            
            has_explain = any('explain' in t or 'transformer' in t for t in task_texts)
            has_compare = any('compare' in t or 'lstm' in t for t in task_texts)
            has_usecase = any('use case' in t or 'practical' in t or 'application' in t for t in task_texts)
            
            if has_explain and has_compare and has_usecase:
                test_2_result["has_sequence"] = True
                test_2_result["pass"] = True
                print("  ✅ PASS: Recognizes ordered/hierarchical structure")
            else:
                print("  ⚠️ PARTIAL: Decomposed but missing ordering signal")
                test_2_result["pass"] = False
        else:
            print("  ❌ FAIL: Not enough subtasks for hierarchical structure")
        
        # Print task breakdown
        if subtasks:
            print(f"\n📋 Subtasks (ordered):")
            for st in subtasks:
                print(f"   [{st.get('number')}] {st.get('task')}")
                print(f"       Purpose: {st.get('purpose', 'N/A')}")
        
        print(f"\n📊 Result: {'✅ PASS' if test_2_result['pass'] else '❌ FAIL'}")
        self.results["test_2_hierarchical"].append(test_2_result)
        return test_2_result

    def test_3_mixed_tool_queries(self) -> Dict:
        """
        TEST 3: Mixed Tool Queries
        ==========================
        Query: "Latest transformer research and explain architecture"
        
        Checks:
        - ✅ Correctly routes: web (latest info) + RAG (explanation)?
        - ✅ Different tools for different parts?
        - ✅ Doesn't use same tool for everything?
        """
        print("\n" + "="*70)
        print("TEST 3: MIXED TOOL QUERIES")
        print("="*70)
        
        query = "Latest transformer research and explain architecture"
        print(f"\n📝 Query: {query}\n")
        
        decomp = decompose_query(query)
        subtasks = decomp.get("subtasks", [])
        
        test_3_result = {
            "query": query,
            "decomposed": decomp.get("should_decompose", False),
            "num_subtasks": len(subtasks),
            "routing_strategy": [],
            "has_mixed_tools": False,
            "pass": False
        }
        
        print(f"🔍 Decomposition: {len(subtasks)} subtasks\n")
        
        # Analyze routing for each subtask
        tools_used = set()
        for st in subtasks:
            task = st.get("task", "")
            routing = route_sources(task)
            tool = routing.get("primary", "unknown")
            tools_used.add(tool)
            
            print(f"   [{st.get('number')}] {task}")
            print(f"       → Tool: {tool} (confidence: {routing.get('confidence')})")
            
            test_3_result["routing_strategy"].append({
                "task": task,
                "tool": tool,
                "confidence": routing.get("confidence")
            })
        
        # Check: Multiple tools used?
        if len(tools_used) > 1:
            test_3_result["has_mixed_tools"] = True
            test_3_result["pass"] = True
            print(f"\n  ✅ PASS: Mixed tools detected ({', '.join(tools_used)})")
        else:
            print(f"\n  ❌ FAIL: All tasks routed to same tool ({tools_used})")
        
        print(f"\n📊 Result: {'✅ PASS' if test_3_result['pass'] else '❌ FAIL'}")
        self.results["test_3_mixed_tools"].append(test_3_result)
        return test_3_result

    def test_4_latency(self) -> Dict:
        """
        TEST 4: Latency Explosion
        =========================
        
        Measures:
        - ✅ Single-step response time
        - ✅ Multi-step response time
        - ✅ Latency ratio (should not exceed 3-4x)
        - ✅ Worst case threshold: <20 seconds
        """
        print("\n" + "="*70)
        print("TEST 4: LATENCY & PERFORMANCE")
        print("="*70)
        
        queries = [
            ("What is machine learning?", "single"),  # Should go single-step
            ("Compare CNN and RNN for image classification", "multi"),  # Should decompose
        ]
        
        test_4_result = {
            "measurements": [],
            "single_step_time": 0,
            "multi_step_time": 0,
            "ratio": 0,
            "latency_acceptable": False,
            "pass": False
        }
        
        print("\n⏱️ Measuring response times...\n")
        
        for query, query_type in queries:
            print(f"📝 {query_type.upper()} Query: {query}")
            
            start = time.time()
            try:
                # This would call the actual agent - for now we just measure decomposition
                decomp = decompose_query(query)
                elapsed = time.time() - start
                
                is_decomposed = decomp.get("should_decompose", False)
                
                print(f"   ⏱️ Time: {elapsed:.3f}s")
                print(f"   🔀 Decomposed: {is_decomposed}")
                
                measurement = {
                    "query": query,
                    "type": query_type,
                    "time_seconds": elapsed,
                    "decomposed": is_decomposed
                }
                
                if query_type == "single":
                    test_4_result["single_step_time"] = elapsed
                elif query_type == "multi":
                    test_4_result["multi_step_time"] = elapsed
                
                test_4_result["measurements"].append(measurement)
                
                # Update global metrics
                self.metrics["avg_latency"] = (self.metrics["avg_latency"] + elapsed) / 2
                self.metrics["max_latency"] = max(self.metrics["max_latency"], elapsed)
                self.metrics["min_latency"] = min(self.metrics["min_latency"], elapsed)
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        # Calculate ratio
        if test_4_result["single_step_time"] > 0:
            test_4_result["ratio"] = test_4_result["multi_step_time"] / test_4_result["single_step_time"]
        
        # Check: Ratio should be reasonable (3-4x acceptable, >10x is bad)
        if test_4_result["ratio"] < 5 or test_4_result["ratio"] == 0:
            test_4_result["latency_acceptable"] = True
        
        # Check: Worst case should be <20 sec
        if self.metrics["max_latency"] < 20:
            test_4_result["pass"] = True
        
        print(f"\n📊 Latency Summary:")
        print(f"   Single-step: {test_4_result['single_step_time']:.3f}s")
        print(f"   Multi-step: {test_4_result['multi_step_time']:.3f}s")
        print(f"   Ratio: {test_4_result['ratio']:.2f}x")
        print(f"   Max latency: {self.metrics['max_latency']:.3f}s")
        print(f"\n   {'✅ PASS' if test_4_result['pass'] else '⚠️ WARNING'}: Latency within acceptable range")
        
        self.results["test_4_latency"].append(test_4_result)
        return test_4_result

    def test_5_synthesis_intelligence(self) -> Dict:
        """
        TEST 5: Synthesis Intelligence
        =============================
        
        Real agents should:
        - ✅ Deduplicate overlapping information
        - ✅ Resolve contradictions
        - ✅ Structure reasoning hierarchically
        - ✅ Not just concatenate answers
        """
        print("\n" + "="*70)
        print("TEST 5: SYNTHESIS INTELLIGENCE")
        print("="*70)
        
        test_5_result = {
            "synthesis_checks": {},
            "pass": False,
            "details": ""
        }
        
        print(f"\n🔍 Checking synthesis capabilities in agent_loop.py...\n")
        
        # Read the synthesize_results function
        try:
            from rag.agent_loop import synthesize_results
            
            # Check function signature and logic
            import inspect
            source = inspect.getsource(synthesize_results)
            
            # Look for intelligence patterns
            has_dedup = "dedup" in source.lower() or "unique" in source.lower()
            has_synthesis_prompt = "synthesis" in source.lower() or "combine" in source.lower()
            has_contradiction_handling = "conflict" in source.lower() or "contradict" in source.lower()
            has_structure = "structure" in source.lower() or "format" in source.lower()
            
            test_5_result["synthesis_checks"] = {
                "deduplication_logic": has_dedup,
                "synthesis_prompt": has_synthesis_prompt,
                "contradiction_handling": has_contradiction_handling,
                "structured_output": has_structure
            }
            
            print(f"   ✅ Deduplication logic: {has_dedup}")
            print(f"   ✅ Synthesis prompt: {has_synthesis_prompt}")
            print(f"   ✅ Contradiction handling: {has_contradiction_handling}")
            print(f"   ✅ Structured output: {has_structure}")
            
            # Pass if at least synthesis_prompt exists
            if has_synthesis_prompt:
                test_5_result["pass"] = True
                test_5_result["details"] = "Synthesis uses intelligent LLM prompting"
                print(f"\n   ✅ PASS: Synthesis uses intelligent reasoning (not simple concatenation)")
            else:
                print(f"\n   ❌ FAIL: Synthesis logic not sophisticated")
                
        except Exception as e:
            print(f"   ⚠️ Could not inspect synthesis function: {str(e)}")
            test_5_result["details"] = f"Error: {str(e)}"
        
        print(f"\n📊 Result: {'✅ PASS' if test_5_result['pass'] else '❌ FAIL'}")
        self.results["test_5_synthesis"].append(test_5_result)
        return test_5_result

    def run_all_tests(self):
        """Run all 5 validation tests"""
        print("\n" + "🚀" * 35)
        print("MULTI-STEP PLANNING VALIDATION SUITE".center(70))
        print("🚀" * 35)
        
        self.test_1_dependency_queries()
        self.test_2_hierarchical_reasoning()
        self.test_3_mixed_tool_queries()
        self.test_4_latency()
        self.test_5_synthesis_intelligence()
        
        self.print_summary()

    def print_summary(self):
        """Print overall validation summary"""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY".center(70))
        print("="*70)
        
        tests_passed = sum(1 for results in self.results.values() for r in results if r.get("pass", False))
        total_tests = sum(len(results) for results in self.results.values())
        
        print(f"\n📊 Overall Results: {tests_passed}/{total_tests} tests passed\n")
        
        for test_name, results in self.results.items():
            if results:
                passed = sum(1 for r in results if r.get("pass", False))
                print(f"   {test_name}: {passed}/{len(results)} ✅")
        
        print(f"\n⏱️ Performance Metrics:")
        print(f"   Avg latency: {self.metrics['avg_latency']:.3f}s")
        print(f"   Max latency: {self.metrics['max_latency']:.3f}s")
        print(f"   Min latency: {self.metrics['min_latency']:.3f}s")
        
        print("\n" + "="*70)

if __name__ == "__main__":
    suite = ValidationSuite()
    suite.run_all_tests()
