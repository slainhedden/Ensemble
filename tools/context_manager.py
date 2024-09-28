from typing import List, Dict, Any
from concurrency.llm_core import llm_core

class ContextManager:
    def __init__(self):
        self.context: List[Dict[str, Any]] = []

    async def add_to_context(self, entry: Dict[str, Any]):
        self.context.append(entry)
        if len(self.context) > 1000:  # Limit to last 1000 entries to manage memory
            self.context = self.context[-1000:]

    async def get_relevant_context(self, query: str, project_overview: str) -> str:
        return await llm_core.get_relevant_context(query, project_overview, self.context)

    async def summarize_for_new_agent(self, agent_role: str, task_description: str, project_overview: str) -> str:
        return await llm_core.summarize_for_new_agent(agent_role, task_description, project_overview, self.context)

context_manager = ContextManager()