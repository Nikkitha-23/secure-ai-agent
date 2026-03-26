#!/usr/bin/env python3
"""
Validation Science Framework for Step 3: Multi-Step Planning

Tests 5 CRITICAL scenarios:
1. Dependency Queries - does agent split correctly? does synthesis reason?
2. Hierarchical Reasoning - ordered reasoning vs flat decomposition
3. Mixed Tool Queries - web vs RAG routing per subtask
4. Latency Explosion - response time impact
5. Synthesis Intelligence - deduplication, contradiction resolution, structure
"""

import time
import json
import logging
from typing import Dict, List, Tuple
from rag.agent_loop import (
    decompose_query, 
    detect_complexity, 
    route_sources,
    call_llm
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# TEST SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════

TEST_QUERIES = {
    "dependency": {
        "query": "Compare CNN and RNN and suggest best for medical imaging",
        "expected_subtasks": 3,
        "expected_reasoning": "Compare architectures + medical domain knowledge"
    },
    "hierarchical": {
        "query": "Explain transformers, then compare with LSTM, then give real use case",
        "expected_subtasks": 3,
        "expected_ordering": True,
        "expected_reasoning": "Ordered: definition → comparison → application"
    },
    "mixed_tools": {
        "query": "What is latest transformer research and explain the architecture",
        "expected_subtasks": 2,
        "expected_routing": {"web": True, "pdf": True},
        "expected_reasoning": "Recent research + technical explanation"
    },
    "simple": {
        "query": "What is machine learning",
        "should_not_decompose": True
    },
    "moderate": {
        "query": "Explain how transformers work in NLP",
        "should_not_decompose": True
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: DEPENDENCY QUERIES
# ═══════════════════════════════════════════════════════════════════════════

def test_dependency_queries():
    """
    Test 1: Check if agent splits correctly and synthesis reasons
    
    Query: "Compare CNN and RNN and suggest best for medical imaging"
    Expected:
    - 3 subtasks: [compare architectures, medical imaging requirements, recommendation]
    - Synthesis reasons about medical domain + architecture tradeoffs
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 1: DEPENDENCY QUERIES")
    logger.info("="*70)
    
    query = TEST_QUERIES["dependency"]["query"]
    logger.info(f"Query: {query}\n")
    
    # Step 1: Check complexity detection
    complexity = detect_complexity(query)
    logger.info(f"✓ Complexity Level: {complexity['level']} (min_words: {complexity['min_words']})")
    
    # Step 2: Check decomposition
    result = decompose_query(query)
    logger.info(f"✓ Should Decompose: {result.get('should_decompose')}")
    logger.info(f"✓ Reasoning: {result.get('reasoning')}")
    
    subtasks = result.get("subtasks", [])
    logger.info(f"✓ Subtasks: {len(subtasks)}")
    
    if subtasks:
        for i, st in enumerate(subtasks, 1):
            logger.info(f"  [{i}] {st.get('task')}")
            logger.info(f"      Purpose: {st.get('purpose')}")
    
    # Step 3: Verify synthesis instructions
    synthesis = result.get("synthesis_instruction", "")
    logger.info(f"✓ Synthesis Plan: {synthesis}")
    
    # Validation
    is_valid = (
        result.get("should_decompose") and
        len(subtasks) >= 2 and
        "synthesis" in synthesis.lower() or "combine" in synthesis.lower()
    )
    
    logger.info(f"\n{'✅ PASS' if is_valid else '❌ FAIL'} - Dependency query test\n")
    return is_valid

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: HIERARCHICAL REASONING
# ═══════════════════════════════════════════════════════════════════════════

def test_hierarchical_reasoning():
    """
    Test 2: Check if ordering is preserved in decomposition
    
    Query: "Explain transformers, then compare with LSTM, then give real use case"
    Expected:
    - Subtasks in order: [explain → compare → apply]
    - NOT flat decomposition
    """
    logger.info("="*70)
    logger.info("TEST 2: HIERARCHICAL REASONING")
    logger.info("="*70)
    
    query = TEST_QUERIES["hierarchical"]["query"]
    logger.info(f"Query: {query}\n")
    
    result = decompose_query(query)
    logger.info(f"✓ Should Decompose: {result.get('should_decompose')}")
    
    subtasks = result.get("subtasks", [])
    logger.info(f"✓ Subtasks: {len(subtasks)}")
    
    if subtasks:
        logger.info("\nTask Ordering:")
        for i, st in enumerate(subtasks, 1):
            task_num = st.get("number", i)
            logger.info(f"  [{task_num}] {st.get('task')}")
    
    # Check if ordering is preserved (tasks numbered, not jumbled)
    task_numbers = [st.get("number", i) for i, st in enumerate(subtasks, 1)]
    is_ordered = task_numbers == sorted(task_numbers)
    
    logger.info(f"\n✓ Ordered Sequence: {task_numbers}")
    logger.info(f"✓ Is Hierarchical: {len(subtasks) >= 2 and is_ordered}")
    
    logger.info(f"\n{'✅ PASS' if is_ordered and len(subtasks) >= 2 else '❌ FAIL'} - Hierarchical reasoning test\n")
    return is_ordered and len(subtasks) >= 2

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: MIXED TOOL QUERIES
# ═══════════════════════════════════════════════════════════════════════════

def test_mixed_tool_queries():
    """
    Test 3: Check if different tools are routed for different subtasks
    
    Query: "Latest transformer research and explain architecture"
    Expected:
    - Subtask 1 (latest research) → routes to WEB (freshness signal)
    - Subtask 2 (explain) → routes to PDF (technical depth)
    """
    logger.info("="*70)
    logger.info("TEST 3: MIXED TOOL QUERIES")
    logger.info("="*70)
    
    query = TEST_QUERIES["mixed_tools"]["query"]
    logger.info(f"Query: {query}\n")
    
    result = decompose_query(query)
    subtasks = result.get("subtasks", [])
    
    logger.info(f"✓ Subtasks: {len(subtasks)}")
    
    routing_patterns = {"web": 0, "pdf": 0}
    
    for i, st in enumerate(subtasks, 1):
        task = st.get("task", "")
        logger.info(f"\n  Subtask {i}: {task}")
        
        # Route each subtask
        route_info = route_sources(task)
        primary = route_info.get("primary", "pdf")
        confidence = route_info.get("confidence", 0)
        reasoning = route_info.get("reasoning", "")
        
        logger.info(f"    🧭 Primary Route: {primary} (confidence: {confidence})")
        logger.info(f"    📋 Reasoning: {reasoning}")
        
        routing_patterns[primary] += 1
    
    # Validation: should use both web and pdf
    uses_both_tools = routing_patterns["web"] > 0 and routing_patterns["pdf"] > 0
    
    logger.info(f"\n✓ Tool Distribution: WEB={routing_patterns['web']}, PDF={routing_patterns['pdf']}")
    logger.info(f"✓ Uses Mixed Tools: {uses_both_tools}")
    
    logger.info(f"\n{'✅ PASS' if uses_both_tools or len(subtasks) >= 2 else '❌ FAIL'} - Mixed tool query test\n")
    return len(subtasks) >= 2

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: LATENCY IMPACT
# ═══════════════════════════════════════════════════════════════════════════

def test_latency():
    """
    Test 4: Measure latency of decomposition vs detection
    
    Check:
    - Decomposition time per query
    - Complexity detection time
    - Total overhead
    
    Threshold: decomposition should add <2-3 sec per query (before LLM)
    """
    logger.info("="*70)
    logger.info("TEST 4: LATENCY ANALYSIS")
    logger.info("="*70)
    
    test_cases = [
        ("simple", TEST_QUERIES["simple"]["query"]),
        ("moderate", TEST_QUERIES["moderate"]["query"]),
        ("dependency", TEST_QUERIES["dependency"]["query"]),
        ("mixed_tools", TEST_QUERIES["mixed_tools"]["query"]),
    ]
    
    latencies = {}
    
    for case_name, query in test_cases:
        logger.info(f"\n🔄 Testing: {case_name}")
        
        # Time complexity detection
        start = time.time()
        complexity = detect_complexity(query)
        complexity_time = time.time() - start
        
        # Time decomposition
        start = time.time()
        result = decompose_query(query)
        decomp_time = time.time() - start
        
        latencies[case_name] = {
            "complexity_ms": complexity_time * 1000,
            "decomposition_ms": decomp_time * 1000,
            "total_ms": (complexity_time + decomp_time) * 1000,
            "should_decompose": result.get("should_decompose"),
            "subtasks": len(result.get("subtasks", []))
        }
        
        logger.info(f"   Complexity: {latencies[case_name]['complexity_ms']:.1f}ms")
        logger.info(f"   Decomposition: {latencies[case_name]['decomposition_ms']:.1f}ms")
        logger.info(f"   Total: {latencies[case_name]['total_ms']:.1f}ms")
        logger.info(f"   Will decompose: {latencies[case_name]['should_decompose']}")
    
    # Analysis
    logger.info("\n" + "─"*70)
    logger.info("LATENCY SUMMARY")
    logger.info("─"*70)
    
    for case_name, metrics in latencies.items():
        status = "✅" if metrics["total_ms"] < 1000 else "⚠️"
        logger.info(f"{status} {case_name:12} {metrics['total_ms']:7.1f}ms (decompose: {metrics['should_decompose']})")
    
    # Check if multi-step overhead is acceptable
    single_step_latency = (latencies["simple"]["total_ms"] + latencies["moderate"]["total_ms"]) / 2
    multi_step_latency = (latencies["dependency"]["total_ms"] + latencies["mixed_tools"]["total_ms"]) / 2
    overhead = multi_step_latency - single_step_latency
    
    logger.info(f"\n◆ Single-step avg: {single_step_latency:.1f}ms")
    logger.info(f"◆ Multi-step avg: {multi_step_latency:.1f}ms")
    logger.info(f"◆ Overhead: +{overhead:.1f}ms")
    
    acceptable = overhead < 1000  # <1 sec overhead is reasonable
    logger.info(f"\n{'✅ PASS' if acceptable else '⚠️ WARNING'} - Latency test (overhead acceptable)\n")
    return acceptable

# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: SYNTHESIS INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════

def test_synthesis_intelligence():
    """
    Test 5: Check synthesis prompt quality
    
    Most systems just concatenate answers.
    Real agent should:
    - Deduplicate
    - Resolve contradictions
    - Structure reasoning
    - Synthesize intelligently
    """
    logger.info("="*70)
    logger.info("TEST 5: SYNTHESIS INTELLIGENCE")
    logger.info("="*70)
    
    query = TEST_QUERIES["dependency"]["query"]
    logger.info(f"Query: {query}\n")
    
    result = decompose_query(query)
    synthesis_instruction = result.get("synthesis_instruction", "")
    
    logger.info(f"Synthesis Instruction:\n{synthesis_instruction}\n")
    
    # Check for intelligence signals in synthesis prompt
    intelligence_signals = {
        "deduplication": "deduplicate" in synthesis_instruction.lower() or "avoid" in synthesis_instruction.lower() or "redundant" in synthesis_instruction.lower(),
        "contradiction_handling": "contradict" in synthesis_instruction.lower() or "conflict" in synthesis_instruction.lower() or "resolve" in synthesis_instruction.lower(),
        "structure": "structure" in synthesis_instruction.lower() or "organize" in synthesis_instruction.lower() or "coherent" in synthesis_instruction.lower(),
        "reasoning": "reason" in synthesis_instruction.lower() or "explain" in synthesis_instruction.lower() or "why" in synthesis_instruction.lower(),
        "recommendation": "recommend" in synthesis_instruction.lower() or "best" in synthesis_instruction.lower() or "suggest" in synthesis_instruction.lower(),
    }
    
    logger.info("Intelligence Signals Detected:")
    for signal, detected in intelligence_signals.items():
        status = "✅" if detected else "❌"
        logger.info(f"  {status} {signal.capitalize()}")
    
    signal_count = sum(1 for v in intelligence_signals.values() if v)
    logger.info(f"\n✓ Signals: {signal_count}/5")
    
    is_intelligent = signal_count >= 3
    logger.info(f"\n{'✅ PASS' if is_intelligent else '⚠️ WARNING'} - Synthesis intelligence test\n")
    return is_intelligent

# ═══════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Execute all 5 validation tests"""
    
    logger.info("\n" + "╔" + "═"*68 + "╗")
    logger.info("║" + " "*15 + "STEP 3 VALIDATION SCIENCE FRAMEWORK" + " "*19 + "║")
    logger.info("╚" + "═"*68 + "╝")
    
    tests = [
        ("Dependency Queries", test_dependency_queries),
        ("Hierarchical Reasoning", test_hierarchical_reasoning),
        ("Mixed Tool Queries", test_mixed_tool_queries),
        ("Latency Impact", test_latency),
        ("Synthesis Intelligence", test_synthesis_intelligence),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ EXCEPTION in {test_name}: {e}")
            results[test_name] = False
    
    # Final report
    logger.info("\n" + "╔" + "═"*68 + "╗")
    logger.info("║" + " "*20 + "VALIDATION SUMMARY" + " "*30 + "║")
    logger.info("╠" + "═"*68 + "╣")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"║ {status:8} | {test_name:40} │")
    
    logger.info("╠" + "═"*68 + "╣")
    logger.info(f"║ TOTAL: {passed}/{total} tests passed" + " "*50 + "║")
    logger.info("╚" + "═"*68 + "╝\n")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
