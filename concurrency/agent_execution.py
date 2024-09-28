import asyncio
from typing import List, Dict, Any
from agents.agent_init import Agent
from ensemble.swarmify import Swarm
from concurrency.llm_core import llm_core
from ensemble.swarmify import initialize_swarm, run_swarm

class AgentExecutor:
    def __init__(self, swarm: Swarm):
        self.swarm = swarm

    async def execute_agent_task(self, agent: Agent):
        if agent.activated and agent.current_task:
            collaboration_group = self.get_collaboration_group(agent)
            if collaboration_group:
                await self.execute_collaborative_task(collaboration_group)
            else:
                await self.execute_individual_task(agent)

    async def execute_collaborative_task(self, collaboration_group: List[Agent]):
        task = collaboration_group[0].current_task  # Assuming all agents in the group have the same task
        prompt = f"As a collaborative group, how would you approach the task: {task['description']}?"
        
        # Use the chat environment for collaboration
        if collaboration_group[0].chat_env:
            await collaboration_group[0].chat_env.broadcast_message("System", f"Collaborative task: {prompt}")
            for agent in collaboration_group:
                response = await llm_core.generate_response(agent.name, f"My thoughts on the task: {prompt}", agent.thread_id)
                await agent.send_message(response)
        
        # Combine the responses from the chat history
        combined_response = "\n".join([msg["message"] for msg in collaboration_group[0].chat_env.chat_history[-len(collaboration_group):]])
        
        for agent in collaboration_group:
            agent.code_output = combined_response
        
        print(f"Collaboration group completed task: {task['description']}")
        print(f"Combined output: {combined_response[:200]}...")  # Print first 200 characters

    async def execute_individual_task(self, agent: Agent):
        prompt = f"As a {agent.role} specialist, how would you approach the task: {agent.current_task['description']}?"
        response = await llm_core.generate_response(agent.name, prompt, agent.thread_id)
        agent.code_output = response
        
        if agent.chat_env:
            await agent.send_message(f"I've completed the task: {agent.current_task['description']}. Here's my approach: {response[:200]}...")
        
        print(f"{agent.name} completed task: {agent.current_task['description']}")
        print(f"Output: {agent.code_output[:200]}...")  # Print first 200 characters

    def get_collaboration_group(self, agent: Agent) -> List[Agent]:
        for group in self.swarm.collaboration_groups:
            if agent in group:
                return group
        return []

    async def run_agents_concurrently(self):
        tasks = [self.execute_agent_task(agent) for agent in self.swarm.agents if agent.activated and agent.current_task]
        await asyncio.gather(*tasks)

    async def run(self):
        await self.run_agents_concurrently()

    async def plan_and_create_agents(self, swarm: Swarm, goal: str):
        await swarm.generate_tasks_and_agents(goal)
        print("Planning complete. Created agents:")
        for agent in swarm.agents:
            print(f"- {agent.name} ({agent.role})")
        print("\nTasks:")
        for task in swarm.tasks:
            print(f"- {task['description']} (Role: {task['role']}, Priority: {task['priority']})")