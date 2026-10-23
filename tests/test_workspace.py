"""
tests/test_workspace.py — Comprehensive Workspace & Task State Test Suite
--------------------------------------------------------------------------
Validates:
  1. Creating a workspace for a simple query.
  2. Tracking a multi-step task with decomposition & execution plan.
  3. Updating the current step, pending steps, and completed steps.
  4. Recording a failed subtask without losing previous results.
  5. Retrieving task state by task ID.
  6. Preventing access to another session's task (session isolation).
  7. Saving and restoring state after server restart (SQLite durable recovery).
  8. Ensuring existing single-step and multi-step agent workflows still work.
"""

import os
import sys
import time
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.workspace import (
    AgentWorkspace,
    WorkspaceStep,
    SQLiteWorkspaceStorage,
    WorkspaceManager,
    truncate_output,
    MAX_STORED_OUTPUT_LEN,
    get_or_create_workspace
)
from core.agent import run_agent, SecureAgent
from rag.agent_loop import decompose_query


class TestWorkspaceTaskState(unittest.TestCase):

    def setUp(self):
        # Create a unique temporary database file for test isolation
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_workspace.db")
        self.storage = SQLiteWorkspaceStorage(self.db_path)
        self.manager = WorkspaceManager(storage=self.storage)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # ── Test 1: Simple Query Workspace Creation ───────────────────────────────
    def test_01_create_workspace_simple_query(self):
        """1. Creating a workspace for a simple query."""
        query = "What is the capital of France?"
        session_id = "sess_user_101"

        ws = self.manager.create_task(session_id=session_id, query=query)
        self.assertIsNotNone(ws.task_id)
        self.assertEqual(ws.session_id, session_id)
        self.assertEqual(ws.original_query, query)
        self.assertEqual(ws.status, "pending")
        self.assertEqual(len(ws.steps), 0)

        # Transition to running
        ws.start()
        self.assertEqual(ws.status, "running")

        # Add single execution step
        step = ws.add_step(tool="direct_llm", purpose="Answer simple query")
        self.assertEqual(step.step_num, 1)
        self.assertEqual(step.status, "pending")

        step.start()
        self.assertEqual(step.status, "running")

        step.complete("Paris is the capital of France.")
        self.assertEqual(step.status, "completed")
        self.assertIn("Paris", step.result)

        # Complete task
        ws.complete(answer="Paris is the capital of France.", search_type="direct_llm", confidence=0.99)
        self.assertEqual(ws.status, "completed")
        self.assertEqual(len(ws.completed_steps), 1)
        self.assertEqual(len(ws.failed_steps), 0)
        self.assertGreater(ws.completed_at, 0.0)

    # ── Test 2: Multi-step Task Tracking ──────────────────────────────────────
    def test_02_track_multistep_task(self):
        """2. Tracking a multi-step task with decomposition plan."""
        query = "Compare Python and Rust performance, and explain when to use each"
        session_id = "sess_user_202"

        plan = {
            "can_decompose": True,
            "subtasks": [
                {"number": 1, "task": "Evaluate Python execution characteristics", "purpose": "Python analysis"},
                {"number": 2, "task": "Evaluate Rust execution characteristics", "purpose": "Rust analysis"},
                {"number": 3, "task": "Compare throughput and memory footprint", "purpose": "Synthesis comparison"}
            ],
            "synthesis_instruction": "Deduplicate and structure comparison"
        }

        ws = self.manager.create_task(session_id=session_id, query=query, plan=plan)
        ws.start()

        self.assertEqual(ws.status, "running")
        self.assertEqual(len(ws.plan.get("subtasks", [])), 3)

        # Populate steps corresponding to subtasks
        for st in ws.plan["subtasks"]:
            step = ws.add_step(tool=st["task"], purpose=st["purpose"], step_num=st["number"])
            self.assertEqual(step.status, "pending")

        self.assertEqual(len(ws.steps), 3)
        self.assertEqual(len(ws.pending_steps), 3)
        self.assertEqual(len(ws.completed_steps), 0)

        summary = ws.summary()
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["pending"], 3)
        self.assertEqual(summary["completed"], 0)

    # ── Test 3: Updating Current Step and Completed Steps ─────────────────────
    def test_03_update_current_step_and_completed_steps(self):
        """3. Updating the current step and completed steps."""
        ws = self.manager.create_task(session_id="sess_steps", query="Multi-phase operation")

        s1 = ws.add_step(tool="web_search", purpose="Phase 1 search")
        s2 = ws.add_step(tool="calculator", purpose="Phase 2 calculation")
        s3 = ws.add_step(tool="synthesizer", purpose="Phase 3 synthesis")

        # Initially, current step is s1 (first pending)
        self.assertIsNotNone(ws.current_step())
        self.assertEqual(ws.current_step().step_num, 1)

        # Start and complete step 1
        s1.start()
        self.assertEqual(ws.current_step().step_num, 1)
        s1.complete("Found 5 data sources")

        # Now completed_steps has 1, current_step is s2
        self.assertEqual(len(ws.completed_steps()), 1)
        self.assertEqual(len(ws.completed_steps), 1)  # Property access
        self.assertEqual(ws.current_step().step_num, 2)

        # Start and complete step 2
        s2.start()
        s2.complete("Calculated average: 42.0")
        self.assertEqual(len(ws.completed_steps), 2)
        self.assertEqual(ws.current_step().step_num, 3)

        # Complete step 3
        s3.start()
        s3.complete("Final summary generated.")
        self.assertEqual(len(ws.completed_steps), 3)
        self.assertIsNone(ws.current_step())

    # ── Test 4: Record Failed Subtask Without Losing Previous Results ─────────
    def test_04_record_failed_subtask_without_losing_previous_results(self):
        """4. Recording a failed subtask without losing previous results."""
        ws = self.manager.create_task(session_id="sess_fail_test", query="Resilient pipeline")

        # Step 1 succeeds
        s1 = ws.add_step(tool="retriever", purpose="Fetch documents")
        s1.start()
        s1.complete("Document A and Document B successfully retrieved.")

        # Step 2 fails with error details
        s2 = ws.add_step(tool="broken_tool", purpose="Faulty extraction")
        s2.start()
        error_msg = "Connection timeout after 30000ms: remote host unreachable"
        s2.fail(error_msg)

        # Verify step 1 result is NOT lost
        self.assertEqual(len(ws.completed_steps), 1)
        self.assertEqual(ws.completed_steps[0].result, "Document A and Document B successfully retrieved.")

        # Verify step 2 failure was captured
        self.assertEqual(len(ws.failed_steps), 1)
        self.assertEqual(ws.failed_steps[0].step_num, 2)
        self.assertEqual(ws.failed_steps[0].error, error_msg)
        self.assertIn("Error:", ws.failed_steps[0].result)

        # Attempting to complete the task must NOT mark it completed when required steps failed
        ws.complete(answer="Attempted answer", search_type="multi-step")
        self.assertEqual(ws.status, "failed")
        self.assertNotEqual(ws.status, "completed")
        self.assertIn("failed", ws.error.lower())

    # ── Test 5: Retrieve Task State by Task ID ────────────────────────────────
    def test_05_retrieve_task_state_by_task_id(self):
        """5. Retrieving task state by task ID."""
        ws_original = self.manager.create_task(
            session_id="sess_lookup",
            query="Deep analysis of quantum computing"
        )
        s1 = ws_original.add_step(tool="web_search", purpose="Literature survey")
        s1.start()
        s1.complete("Survey completed with 8 papers.")
        ws_original.note("Completed initial literature review")
        ws_original.complete(answer="Quantum supremacy demonstrated in specific algorithms.")

        # Retrieve by task_id
        ws_retrieved = self.manager.get_workspace(ws_original.task_id)
        self.assertIsNotNone(ws_retrieved)
        self.assertEqual(ws_retrieved.task_id, ws_original.task_id)
        self.assertEqual(ws_retrieved.original_query, "Deep analysis of quantum computing")
        self.assertEqual(ws_retrieved.status, "completed")
        self.assertEqual(len(ws_retrieved.steps), 1)
        self.assertEqual(ws_retrieved.steps[0].result, "Survey completed with 8 papers.")
        self.assertIn("Quantum supremacy", ws_retrieved.final_answer)

        # Safe handling for invalid / non-existent task IDs
        self.assertIsNone(self.manager.get_workspace("non_existent_id_99999"))
        self.assertIsNone(self.manager.get_workspace(""))
        self.assertIsNone(self.manager.get_workspace(None))

    # ── Test 6: Preventing Access to Another Session's Task ───────────────────
    def test_06_prevent_access_to_another_sessions_task(self):
        """6. Preventing access to another session's task (session isolation)."""
        # User Alice creates a task
        alice_task = self.manager.create_task(
            session_id="session_alice_secure",
            query="Alice's confidential financial report"
        )
        alice_task.complete(answer="Confidential Q3 figures: $5M revenue")

        # Alice accesses her own task -> allowed
        retrieved_alice = self.manager.get_workspace(alice_task.task_id, session_id="session_alice_secure")
        self.assertIsNotNone(retrieved_alice)
        self.assertEqual(retrieved_alice.task_id, alice_task.task_id)

        # User Bob attempts to access Alice's task -> raises PermissionError
        with self.assertRaises(PermissionError) as context:
            self.manager.get_workspace(alice_task.task_id, session_id="session_bob_unauthorized")

        self.assertIn("Access denied", str(context.exception))
        self.assertIn("does not belong to session", str(context.exception))

    # ── Test 7: Save and Restore State After Server Restart ───────────────────
    def test_07_save_and_restore_state_after_restart(self):
        """7. Saving and restoring state after a restart if durable persistence is implemented."""
        # 1. Simulate active server runtime creating and modifying state
        task_id = "restart_test_task"
        ws_before = self.manager.create_task(
            session_id="sess_persistent",
            query="Long-running data migration",
            task_id=task_id,
            plan={"subtasks": [{"task": "Subtask 1"}, {"task": "Subtask 2"}]}
        )
        ws_before.start()
        step1 = ws_before.add_step(tool="extractor", purpose="Extract rows")
        step1.start()
        step1.complete("Extracted 10,000 records.")

        step2 = ws_before.add_step(tool="loader", purpose="Load to warehouse")
        step2.start()
        step2.fail("Deadlock detected during batch insert.")

        ws_before.note("Encountered deadlock on table users")
        ws_before.fail("Batch insert failure")

        # 2. Simulate complete server shutdown: purge in-memory manager
        del ws_before
        del self.manager
        del self.storage

        # 3. Simulate server restart: create brand new storage and manager pointing to same SQLite DB
        restarted_storage = SQLiteWorkspaceStorage(self.db_path)
        restarted_manager = WorkspaceManager(storage=restarted_storage)

        # 4. Recover task state
        ws_recovered = restarted_manager.get_workspace(task_id)
        self.assertIsNotNone(ws_recovered)
        self.assertEqual(ws_recovered.task_id, task_id)
        self.assertEqual(ws_recovered.session_id, "sess_persistent")
        self.assertEqual(ws_recovered.status, "failed")
        self.assertEqual(len(ws_recovered.steps), 2)
        self.assertEqual(ws_recovered.steps[0].status, "completed")
        self.assertEqual(ws_recovered.steps[0].result, "Extracted 10,000 records.")
        self.assertEqual(ws_recovered.steps[1].status, "failed")
        self.assertEqual(len(ws_recovered.scratchpad), 2)
        self.assertIn("deadlock", ws_recovered.scratchpad[0]["note"].lower())
        self.assertIn("failed", ws_recovered.scratchpad[1]["note"].lower())

    # ── Test 8: Agent Workflow Integration & Output Truncation ────────────────
    def test_08_output_truncation_and_agent_integration(self):
        """8. Ensuring output truncation works and workspace integrates safely."""
        # 8a. Output truncation
        huge_text = "X" * (MAX_STORED_OUTPUT_LEN + 1500)
        truncated = truncate_output(huge_text)
        self.assertLess(len(truncated), len(huge_text))
        self.assertIn("truncated 1500 chars", truncated)

        # 8b. Safe singleton helper
        ws = get_or_create_workspace(session_id="test_sess_singleton", query="Hello World")
        self.assertIsNotNone(ws)
        self.assertEqual(ws.original_query, "Hello World")

        # 8c. Agent execution produces valid task_id
        # We test with a direct simple query (e.g. greeting)
        res = run_agent(query="Hello!", session_id="test_sess_agent")
        self.assertIn("answer", res)
        self.assertIn("task_id", res)
        self.assertIsNotNone(res["task_id"])


if __name__ == "__main__":
    unittest.main()
