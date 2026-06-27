import re
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .core.case_bank import CaseBank
from .core.playbook import PlaybookManager
from .core.planner import Planner
from .core.executor import Executor
from .core.generator import Generator
from .core.reflector import Reflector
from .core.curator import Curator
from .utils.llm import initialize_clients


class ACEMementoRunner:
    """
    Main ACE-Memento Orchestrator.
    Manages the continual learning loop coordinating Planner, Executor, Reflector, Curator,
    evolving both Episodic memory (Case Bank) and Semantic memory (Playbook) simultaneously.
    """

    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        reflector_model: str,
        curator_model: str,
        memory_jsonl_path: str = "./results/case_bank.jsonl",
        max_tokens: int = 4096,
        initial_playbook: Optional[str] = None,
        use_rae: bool = False,
        rae_top_k: int = 10,
        case_bank_top_k: int = 4,
        server_scripts: Optional[List[str]] = None,
        device: str = "cpu"
    ):
        self.api_provider = api_provider
        self.generator_model = generator_model
        self.reflector_model = reflector_model
        self.curator_model = curator_model
        self.max_tokens = max_tokens
        self.use_rae = use_rae
        self.rae_top_k = rae_top_k
        self.case_bank_top_k = case_bank_top_k

        # Initialize clients
        generator_client, reflector_client, curator_client = initialize_clients(api_provider)

        # 1. Playbook Manager (Semantic Memory)
        self.playbook_manager = PlaybookManager(
            initial_playbook=initial_playbook,
            device=device
        )

        # 2. Case Bank (Episodic Memory)
        self.case_bank = CaseBank(
            memory_jsonl_path=memory_jsonl_path,
            top_k=case_bank_top_k,
            device=device
        )

        # 3. Core agents
        self.planner = Planner(generator_client, api_provider, generator_model, max_tokens)
        self.executor = Executor(generator_client, api_provider, generator_model, max_tokens, server_scripts)
        self.generator = Generator(self.planner, self.executor)
        
        self.reflector = Reflector(reflector_client, api_provider, reflector_model, max_tokens)
        self.curator = Curator(curator_client, api_provider, curator_model, max_tokens)

        self.next_global_id = 1
        self._recompute_next_global_id()

    def _recompute_next_global_id(self) -> None:
        """Find the next ID to assign to playbook bullets."""
        max_id = 0
        for b in self.playbook_manager.bullets:
            id_match = re.search(r'-(\d+)$', b['id'])
            if id_match:
                num = int(id_match.group(1))
                max_id = max(max_id, num)
        self.next_global_id = max_id + 1

    def run(
        self,
        mode: str,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor: Any = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper around run_async to execute the runner loop.
        """
        return asyncio.run(self.run_async(mode, train_samples, val_samples, test_samples, data_processor, config))

    async def run_async(
        self,
        mode: str,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor: Any = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main run entry point for offline/online training or evaluation.
        """
        config = config or {}
        num_epochs = config.get("num_epochs", 1)
        max_num_rounds = config.get("max_num_rounds", 3)
        token_budget = config.get("playbook_token_budget", 80000)
        save_dir = config.get("save_dir", "./results")
        
        # Connect executor stdio servers
        await self.executor.connect_mcp_servers()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"ace_memento_{timestamp}_{mode}"
        run_path = os.path.join(save_dir, run_folder)
        os.makedirs(run_path, exist_ok=True)
        log_dir = os.path.join(run_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        results = {}

        if mode == "offline":
            print(f"--- Starting Offline Training Epochs={num_epochs} ---")
            train_results = []
            
            for epoch in range(1, num_epochs + 1):
                print(f"--- Epoch {epoch}/{num_epochs} ---")
                
                for step, sample in enumerate(train_samples or [], 1):
                    print(f"\n--- Train Step {step}/{len(train_samples)} ---")
                    
                    # 1. Retrieve dual memory contexts
                    query = sample.get("question", "")
                    context = sample.get("context", "")
                    target = sample.get("target", "")

                    retrieved_cases = self.case_bank.retrieve_cases(query)
                    cases_text = self.case_bank.format_cases_for_prompt(retrieved_cases)
                    
                    playbook = self.playbook_manager.retrieve_bullets(query, self.rae_top_k) if self.use_rae else self.playbook_manager.playbook

                    # 2. Run generator (Planner + Executor)
                    final_answer, bullet_ids_used, trajectory = await self.generator.generate(
                        question=query,
                        playbook=playbook,
                        cases_text=cases_text,
                        context=context,
                        use_json_mode=config.get("json_mode", False),
                        call_id=f"train_e{epoch}_s{step}",
                        log_dir=log_dir
                    )

                    # 3. Evaluate accuracy (reward)
                    is_correct = data_processor.answer_is_correct(final_answer, target)
                    reward = 1 if is_correct else 0
                    print(f"Predicted answer: {final_answer} | Target: {target} | Correct: {is_correct}")

                    # 4. Write case to episodic memory (Memento CASE WRITE)
                    self.case_bank.add_case(query, trajectory["plan_json"], reward)

                    # 5. Reflect and Curate (ACE context engineering)
                    trajectory_str = json.dumps(trajectory, indent=2)
                    bullets_used_str = "\n".join([b["original_line"] for b in self.playbook_manager.bullets if b["id"] in bullet_ids_used])
                    
                    if is_correct:
                        # Correct: reinforce helpful bullets
                        _, bullet_tags, _ = self.reflector.reflect(
                            question=query,
                            trajectory_str=trajectory_str,
                            predicted_answer=final_answer,
                            ground_truth=target,
                            environment_feedback="Predicted answer matches ground truth",
                            bullets_used_str=bullets_used_str,
                            use_ground_truth=True,
                            call_id=f"reflect_s{step}",
                            log_dir=log_dir
                        )
                        # Apply updates to bullet counts
                        updated_playbook = self.curator.update_bullet_counts(self.playbook_manager.playbook, bullet_tags)
                        self.playbook_manager.update_playbook(updated_playbook)
                    else:
                        # Incorrect: run reflection rounds
                        reflection = ""
                        for r in range(max_num_rounds):
                            # Retrieve negative cases for analogical context
                            neg_cases = [c for c in retrieved_cases if c.get("reward") == 0]
                            neg_text = self.case_bank.format_cases_for_prompt(neg_cases)
                            
                            reflection, bullet_tags, _ = self.reflector.reflect(
                                question=query,
                                trajectory_str=trajectory_str,
                                predicted_answer=final_answer,
                                ground_truth=target,
                                environment_feedback="Predicted answer does not match ground truth",
                                bullets_used_str=bullets_used_str,
                                analogical_context=neg_text,
                                use_ground_truth=True,
                                call_id=f"reflect_s{step}_r{r}",
                                log_dir=log_dir
                            )

                            # Tag and reinforce counts
                            updated_playbook = self.curator.update_bullet_counts(self.playbook_manager.playbook, bullet_tags)
                            self.playbook_manager.update_playbook(updated_playbook)

                            # Try to regenerate
                            final_answer, bullet_ids_used, trajectory = await self.generator.generate(
                                question=query,
                                playbook=self.playbook_manager.retrieve_bullets(query, self.rae_top_k) if self.use_rae else self.playbook_manager.playbook,
                                cases_text=cases_text,
                                context=context,
                                use_json_mode=config.get("json_mode", False),
                                call_id=f"train_e{epoch}_s{step}_r{r}",
                                log_dir=log_dir
                            )
                            if data_processor.answer_is_correct(final_answer, target):
                                print(f"Corrected reasoning on round {r}!")
                                break

                        # Curate: evolve semantic playbook rules
                        stats = self.curator.get_playbook_stats(self.playbook_manager.playbook)
                        updated_playbook, self.next_global_id, operations, _ = self.curator.curate(
                            current_playbook=self.playbook_manager.playbook,
                            recent_reflection=reflection,
                            question_context=context,
                            current_step=step,
                            total_samples=len(train_samples),
                            token_budget=token_budget,
                            playbook_stats=stats,
                            call_id=f"curate_s{step}",
                            log_dir=log_dir,
                            next_global_id=self.next_global_id
                        )
                        self.playbook_manager.update_playbook(updated_playbook)

                # Save intermediate playbooks
                epoch_playbook_path = os.path.join(run_path, f"playbook_epoch_{epoch}.txt")
                with open(epoch_playbook_path, "w", encoding="utf-8") as f:
                    f.write(self.playbook_manager.playbook)

            # Save final playbook and case bank
            final_playbook_path = os.path.join(run_path, "final_playbook.txt")
            with open(final_playbook_path, "w", encoding="utf-8") as f:
                f.write(self.playbook_manager.playbook)
            
            results["training"] = "completed"
            print(f"[ACEMementoRunner] Run complete. Results saved to {run_path}")

        elif mode == "eval_only":
            print(f"--- Starting Evaluation on {len(test_samples or [])} samples ---")
            correct = 0
            for step, sample in enumerate(test_samples or [], 1):
                query = sample.get("question", "")
                context = sample.get("context", "")
                target = sample.get("target", "")

                retrieved_cases = self.case_bank.retrieve_cases(query)
                cases_text = self.case_bank.format_cases_for_prompt(retrieved_cases)
                playbook = self.playbook_manager.retrieve_bullets(query, self.rae_top_k) if self.use_rae else self.playbook_manager.playbook

                final_answer, _, _ = await self.generator.generate(
                    question=query,
                    playbook=playbook,
                    cases_text=cases_text,
                    context=context,
                    use_json_mode=config.get("json_mode", False),
                    call_id=f"eval_s{step}",
                    log_dir=log_dir
                )

                is_correct = data_processor.answer_is_correct(final_answer, target)
                if is_correct:
                    correct += 1
                print(f"Eval {step}: Pred={final_answer} | Target={target} | Correct={is_correct}")

            accuracy = correct / len(test_samples) if test_samples else 0.0
            print(f"Evaluation Accuracy: {accuracy:.4f}")
            results["accuracy"] = accuracy

        await self.executor.cleanup()
        return results
