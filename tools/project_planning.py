from typing import List, Dict, Any

class ProjectPlanner:
    def __init__(self):
        self.agents: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []

    def add_agent(self, name: str, role: str, specialties: List[str]):
        agent = {"name": name, "role": role, "specialties": specialties}
        self.agents.append(agent)
        return f"Added agent: {name} ({role})"

    def add_task(self, description: str, role: str, priority: int, dependencies: List[str] = None):
        task = {
            "description": description,
            "role": role,
            "priority": priority,
            "dependencies": dependencies or []
        }
        self.tasks.append(task)
        return f"Added task: {description} (Priority: {priority})"

    def get_agents(self) -> List[Dict[str, Any]]:
        return self.agents

    def get_tasks(self) -> List[Dict[str, Any]]:
        return self.tasks

    def clear(self):
        self.agents = []
        self.tasks = []
        return "Cleared all agents and tasks"

planner = ProjectPlanner()