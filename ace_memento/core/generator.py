import json
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from .planner import Planner
from .executor import Executor
from ..utils.formatting import strip_fences, extract_json_from_text


class Generator:
    """
    Coordinating Generator agent (combining Meta-Planner and Executor).
    Processes user query and subtasks sequentially.
    """

    def __init__(self, planner: Planner, executor: Executor, max_cycles: int = 3):
        self.planner = planner
        self.executor = executor
        self.max_cycles = max_cycles

    async def generate(
        self,
        question: str,
        playbook: str,
        cases_text: str,
        context: str = "",
        use_json_mode: bool = False,
        call_id: str = "gen",
        log_dir: Optional[str] = None
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Generate answer for a query using Plan-Execute cycles.
        
        Returns:
            Tuple of:
              - final_answer (str)
              - bullet_ids_used (list of str)
              - trajectory (dict with all traces)
        """
        shared_history: List[Dict[str, str]] = []
        shared_history.append({"role": "user", "content": f"{question}\nContext: {context}"})

        meta_trace: List[Dict[str, Any]] = []
        executor_trace: List[Dict[str, Any]] = []
        tool_history: List[Dict[str, Any]] = []
        
        bullet_ids_used: List[str] = []
        final_answer = ""
        latest_plan_json = ""

        for cycle in range(self.max_cycles):
            # Format history for planner prompt
            history_lines = []
            for msg in shared_history:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    history_lines.append(f"[User]: {content}")
                elif role == "assistant":
                    history_lines.append(f"[Planner]: {content}")
            history_text = "\n".join(history_lines)

            # 1. Planner generates a plan
            planner_response, call_info = self.planner.plan(
                question=question,
                playbook=playbook,
                cases_text=cases_text,
                history_text=history_text,
                context=context,
                use_json_mode=use_json_mode,
                call_id=f"{call_id}_cycle_{cycle}",
                log_dir=log_dir
            )

            meta_trace.append({
                "cycle": cycle,
                "prompt": call_info.get("prompt", ""),
                "response": planner_response
            })

            shared_history.append({"role": "assistant", "content": planner_response})

            # Check if planner provided final answer
            if "FINAL ANSWER:" in planner_response:
                parts = planner_response.split("FINAL ANSWER:", 1)
                final_answer = parts[1].strip()
                break

            # Try to extract plan JSON
            try:
                plan_data = extract_json_from_text(planner_response)
                if plan_data and "plan" in plan_data:
                    tasks = plan_data["plan"]
                    latest_plan_json = json.dumps(plan_data)
                    
                    # Accumulate used bullet IDs from the planner output
                    if "bullet_ids" in plan_data:
                        for bid in plan_data["bullet_ids"]:
                            if bid not in bullet_ids_used:
                                bullet_ids_used.append(bid)
                else:
                    raise ValueError("No plan found in JSON structure")
            except Exception as e:
                # If plan parsing fails, return response directly
                final_answer = f"[Planner error] {e}: {planner_response}"
                break

            # 2. Executor executes each subtask
            for task in tasks:
                task_id = task.get("id", 1)
                task_desc = f"Task {task_id}: {task.get('description', '')}"

                # Run executor (asynchronous due to tool sessions and asyncio)
                exec_output, steps, tool_calls = await self.executor.execute_task(
                    task_desc=task_desc,
                    history=shared_history,
                    call_id=f"{call_id}_exec_t{task_id}",
                    log_dir=log_dir
                )

                executor_trace.extend(steps)
                tool_history.extend(tool_calls)

                # Append execution outcome back to meta-planner history
                outcome_str = f"Task {task_id} result: {exec_output}"
                shared_history.append({"role": "assistant", "content": outcome_str})

        if not final_answer:
            final_answer = planner_response.strip()

        # Build complete trajectory trace
        trajectory = {
            "question": question,
            "context": context,
            "final_answer": final_answer,
            "plan_json": latest_plan_json,
            "meta_trace": meta_trace,
            "executor_trace": executor_trace,
            "tool_history": tool_history,
            "bullet_ids_used": bullet_ids_used
        }

        return final_answer, bullet_ids_used, trajectory
