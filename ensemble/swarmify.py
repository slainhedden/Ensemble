import asyncio
from typing import List, Dict, Any, Tuple, Optional
from agents.agent_init import Agent
from tools.rag_utils import RAG
from concurrency.llm_core import llm_core
import re
from tools.file_operations import FileOperations
from interaction.chat_environment import ChatEnvironment, initialize_chat_environment
import logging

logger = logging.getLogger(__name__)
logging.getLogger("openai").disabled = True
logging.getLogger("httpx").disabled = True

class TaskGenerator:
    @staticmethod
    async def generate_tasks(goal: str, project_overview: str) -> List[Dict[str, Any]]:
        prompt = f"""Given the project goal: {goal}
And the project overview: {project_overview}

Generate a list of tasks needed to complete this project. Each task should have:
1. A descriptive name
2. The role of the agent who should perform it
3. A priority (1-5, where 1 is highest priority)
4. A unique ID
5. A list of dependency task IDs (can be empty)

Format the output as a JSON list of task objects."""

        response = await llm_core.gemini_generate_content(prompt)
        try:
            tasks = eval(response)  # Convert the string representation to a list of dictionaries
            logger.info(f"Generated {len(tasks)} tasks")
            return tasks
        except Exception as e:
            logger.error(f"Error parsing tasks: {e}")
            logger.error(f"Raw response: {response}")
            return []

class Swarm:
    def __init__(self):
        self.agents: List[Agent] = []
        self.tasks: List[Dict[str, Any]] = []
        self.completed_tasks: List[Dict[str, Any]] = []
        self.project_overview: str = ""
        self.file_ops = FileOperations()
        try:
            self.shared_rag = RAG()
        except Exception as e:
            logger.error(f"Error initializing RAG: {str(e)}")
            self.shared_rag = None
        self.chat_env: Optional['ChatEnvironment'] = None

    async def add_agent(self, agent: Agent, task_description: str):
        await agent.initialize_with_context(task_description, self.project_overview)
        self.agents.append(agent)
        if self.shared_rag:
            agent.rag = self.shared_rag  # Use the shared RAG for all agents
        logger.info(f"Added agent: {agent.name} ({agent.role})")

    async def generate_tasks_and_agents(self, goal: str):
        project_manager = self.get_agent_by_role("Project Manager")
        if project_manager:
            prompt = f"""
            As the Project Manager, create a task list and determine the necessary agents for the project.
            Project Goal: {goal}
            Project Overview: {self.project_overview}

            Create 3-5 tasks and 2-3 agents. For each task, provide:
            1. Description
            2. Role responsible
            3. Priority (1 being highest)
            4. Unique task ID (e.g., T1, T2, etc.)
            5. Dependencies (list of task IDs that must be completed before this task, or 'None' if no dependencies)

            For each agent, provide:
            1. Name
            2. Role
            3. Specialties (comma-separated list)

            Format your response as follows:
            Tasks:
            1. [Task description] | [Role responsible] | Priority: [Priority number] | ID: [Task ID] | Dependencies: [Dependent Task IDs or None]
            2. ...

            Agents:
            1. [Name] | [Role] | [Specialty1, Specialty2, ...]
            2. ...
            """
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await llm_core.generate_response(project_manager.name, prompt)
                    
                    tasks_section = re.search(r"Tasks:(.*?)(?:Agents:|$)", response, re.DOTALL)
                    agents_section = re.search(r"Agents:(.*?)$", response, re.DOTALL)
                    
                    if tasks_section:
                        tasks_text = tasks_section.group(1).strip()
                        tasks = self._parse_tasks(tasks_text)
                        for task in tasks:
                            await self.add_task(task)
                    
                    if agents_section:
                        agents_text = agents_section.group(1).strip()
                        agents = self._parse_agents(agents_text)
                        for agent_info in agents:
                            agent = Agent(agent_info['name'], agent_info['role'], agent_info['specialties'])
                            await self.add_agent(agent, f"You are responsible for tasks related to {agent_info['role']}")
                    
                    print("Planning complete. Created agents:")
                    for agent in self.agents:
                        print(f"- {agent.name} ({agent.role})")
                    
                    print("\nTasks:")
                    for task in self.tasks:
                        print(f"- {task['description']} (Priority: {task['priority']}, Role: {task['role']}, ID: {task['id']}, Dependencies: {task['dependencies']})")
                    break  # If successful, break out of the retry loop
                
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error("Max retries reached. Unable to generate tasks and agents.")
                        raise
                    await asyncio.sleep(1)  # Wait a bit before retrying
        else:
            print("Error: Project Manager not found.")

    async def add_task(self, task: Dict[str, Any]):
        self.tasks.append(task)
        logger.info(f"Added task: {task['description']} (Role: {task['role']}, Priority: {task['priority']})")
        if self.shared_rag:
            await self.shared_rag.upload_data([f"New task: {task['description']}"])

    async def dynamic_task_prioritization(self):
        for task in self.tasks:
            if not task.get('dynamic_priority'):
                task['dynamic_priority'] = task['priority']
            
            # Increase priority for tasks that have been waiting longer
            task['dynamic_priority'] += 0.1
            
            # Decrease priority for tasks with many dependencies not yet completed
            incomplete_dependencies = sum(1 for dep in task['dependencies'] if not self.is_task_completed(dep))
            task['dynamic_priority'] -= 0.05 * incomplete_dependencies

    async def collaborative_task_solving(self, task: Dict[str, Any]):
        suitable_agents = [a for a in self.agents if not a.current_task and task['role'] in a.specialties]
        if len(suitable_agents) >= 2:
            collaboration_group = suitable_agents[:2]  # Select two agents for collaboration
            self.collaboration_groups.append(collaboration_group)
            
            for agent in collaboration_group:
                await agent.assign_task(task)
            
            # Collaborative problem-solving
            solution = await self.collaborative_problem_solving(collaboration_group, task)
            
            # Update task with collaborative solution
            task['collaborative_solution'] = solution

    async def collaborative_problem_solving(self, agents: List[Agent], task: Dict[str, Any]) -> str:
        prompt = f"Collaborate on solving the task: {task['description']}. Each agent should contribute their expertise."
        solutions = await asyncio.gather(*[agent.ask_question(prompt) for agent in agents])
        
        # Combine solutions (this can be more sophisticated)
        combined_solution = "\n".join(solutions)
        
        return combined_solution

    async def agent_specialization_evolution(self):
        for agent in self.agents:
            if agent.completed_tasks:
                new_specialty = await self.determine_new_specialty(agent)
                if new_specialty and new_specialty not in agent.specialties:
                    agent.specialties.append(new_specialty)
                    print(f"{agent.name} has gained a new specialty: {new_specialty}")

    async def determine_new_specialty(self, agent: Agent) -> str:
        task_descriptions = [task['description'] for task in agent.completed_tasks]
        prompt = f"Based on these completed tasks: {task_descriptions}, suggest a new specialty for the agent."
        return await llm_core.generate_response(agent.name, prompt)

    async def inter_agent_knowledge_sharing(self):
        for agent in self.agents:
            knowledge_to_share = agent.get_shareable_knowledge()
            if knowledge_to_share and self.shared_rag:
                await self.shared_rag.upload_data([knowledge_to_share])
                print(f"{agent.name} shared knowledge with the swarm.")

    async def adaptive_swarm_sizing(self):
        workload = len(self.tasks)
        current_agents = len(self.agents)
        
        if workload > current_agents * 2:  # If there are more than twice as many tasks as agents
            new_agent = await self.create_new_agent()
            await self.add_agent(new_agent)
            print(f"New agent {new_agent.name} added to the swarm due to high workload.")
        elif workload < current_agents // 2:  # If there are less than half as many tasks as agents
            agent_to_remove = self.select_agent_to_remove()
            self.agents.remove(agent_to_remove)
            print(f"Agent {agent_to_remove.name} removed from the swarm due to low workload.")

    async def create_new_agent(self) -> Agent:
        # Logic to create a new agent based on current needs
        new_role = await self.determine_needed_role()
        new_name = f"Agent_{len(self.agents) + 1}"
        return Agent(new_name, new_role, [])

    async def determine_needed_role(self) -> str:
        # Analyze current tasks and agent roles to determine the most needed role
        current_roles = [agent.role for agent in self.agents]
        task_roles = [task['role'] for task in self.tasks]
        prompt = f"Given the current roles {current_roles} and required task roles {task_roles}, what new role is most needed?"
        return await llm_core.generate_response("Swarm", prompt)

    def select_agent_to_remove(self) -> Agent:
        # Select the agent with the least completed tasks
        return min(self.agents, key=lambda a: len(a.completed_tasks))

    async def allocate_tasks(self):
        await self.dynamic_task_prioritization()
        sorted_tasks = sorted(self.tasks, key=lambda x: x['dynamic_priority'], reverse=True)
        
        for task in sorted_tasks:
            if not task.get('assigned'):
                if task.get('collaborative', False):
                    await self.collaborative_task_solving(task)
                else:
                    suitable_agents = [a for a in self.agents if not a.current_task and task['role'] in a.specialties]
                    if suitable_agents:
                        agent = min(suitable_agents, key=lambda a: len(a.memory))
                        await agent.assign_task(task)
                        task['assigned'] = True

        await self.agent_specialization_evolution()
        await self.inter_agent_knowledge_sharing()
        await self.adaptive_swarm_sizing()

    async def establish_collaborations(self):
        for i, agent in enumerate(self.agents):
            for other_agent in self.agents[i+1:]:
                agent.add_collaborator(other_agent)
                other_agent.add_collaborator(agent)

    def get_agent_by_name(self, name: str) -> Agent:
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    async def activate_agents(self):
        activation_rounds = max(agent.activation_order.get(agent.role, 0) for agent in self.agents)
        for round in range(1, activation_rounds + 1):
            for agent in self.agents:
                if not agent.activated and agent.can_activate():
                    agent.activate()
            await asyncio.sleep(1)  # Give a short delay between activation rounds

    def get_agent_by_role(self, role: str) -> Agent:
        for agent in self.agents:
            if agent.role == role:
                return agent
        return None

    async def agent_communication(self, sender: Agent, receiver: Agent, message: str) -> str:
        if sender.activated and receiver.activated:
            response = await receiver.ask_question(message)
            print(f"{sender.name} to {receiver.name}: {message}")
            print(f"{receiver.name} response: {response[:200]}...")  # Print first 200 characters
            return response
        else:
            return "Error: One or both agents are not activated."

    async def ask_all_agents(self, question: str) -> Dict[str, str]:
        responses = {}
        for agent in self.agents:
            if agent.activated:
                response = await agent.ask_question(question)
                responses[agent.name] = response
        return responses

    def _parse_tasks(self, tasks_text: str) -> List[Dict[str, Any]]:
        tasks = []
        for line in tasks_text.split('\n'):
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 5:
                    try:
                        priority = int(parts[2].split(':')[-1].strip())
                        task_id = parts[3].split(':')[-1].strip()
                        dependencies = parts[4].split(':')[-1].strip()
                        dependencies = [] if dependencies.lower() == 'none' else [dep.strip() for dep in dependencies.split(',')]
                        task = {
                            'description': parts[0],
                            'role': parts[1],
                            'priority': priority,
                            'id': task_id,
                            'dependencies': dependencies,
                            'assigned': False
                        }
                        tasks.append(task)
                    except ValueError as e:
                        logger.warning(f"Error parsing task: {e}. Skipping this task.")
        return tasks

    def _parse_agents(self, agents_text: str) -> List[Dict[str, Any]]:
        agents = []
        for line in agents_text.split('\n'):
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 3:
                    agent = {
                        'name': parts[0],
                        'role': parts[1],
                        'specialties': [s.strip() for s in parts[2].split(',')]
                    }
                    agents.append(agent)
        return agents

    def is_task_completed(self, task_id: str) -> bool:
        return any(task['id'] == task_id for task in self.completed_tasks)

    async def start_chat(self):
        if self.chat_env:
            await self.chat_env.start_chat()
        else:
            print("Chat environment not initialized. Please call initialize_chat_environment() first.")

    async def run_iteration(self):
        results = []
        for agent in self.agents:
            if not agent.current_task:
                task = self.get_next_task_for_agent(agent)
                if task:
                    agent.current_task = task
                    logger.info(f"Assigned task to {agent.name}: {task['description']}")

            if agent.current_task:
                try:
                    result = await agent.execute_task(agent.current_task)
                    results.append(result)
                    logger.info(f"Task completed by {agent.name}: {agent.current_task['description']}")
                    self.completed_tasks.append(agent.current_task)
                    self.tasks.remove(agent.current_task)
                    agent.current_task = None
                except Exception as e:
                    logger.error(f"Error executing task for {agent.name}: {str(e)}")
                finally:
                    agent.current_task = None
            else:
                logger.debug(f"{agent.name} has no current task")

        await self.dynamic_task_prioritization()
        await self.allocate_tasks()
        await self.agent_specialization_evolution()
        await self.inter_agent_knowledge_sharing()
        await self.adaptive_swarm_sizing()

        return results

    def get_next_task_for_agent(self, agent: Agent) -> Optional[Dict[str, Any]]:
        available_tasks = [task for task in self.tasks if task['role'] == agent.role and not task.get('assigned', False)]
        if available_tasks:
            return min(available_tasks, key=lambda x: x['priority'])
        return None

    async def initialize_chat_environment(self):
        self.chat_env = await initialize_chat_environment(self)
        for agent in self.agents:
            agent.connect_to_chat_environment(self.chat_env)
        logger.info("Chat environment initialized")

    async def start_chat(self):
        if self.chat_env:
            await self.chat_env.start_chat()
        else:
            logger.warning("Chat environment not initialized.")

async def initialize_swarm(goal: str, project_overview: str) -> Swarm:
    swarm = Swarm()
    swarm.project_overview = project_overview
    project_manager = Agent("ProjectManagerBot", "Project Manager", ["planning", "coordination"])
    await swarm.add_agent(project_manager, "Planning and coordinating the project")
    await swarm.generate_tasks_and_agents(goal)
    await swarm.initialize_chat_environment()
    logger.info("Swarm initialized")
    return swarm

async def run_swarm(swarm: Swarm, iterations: int = 1):
    for i in range(iterations):
        logger.info(f"Starting iteration {i+1}")
        results = await swarm.run_iteration()
        for result in results:
            print(result)
        
        print("\nRemaining tasks:")
        for task in swarm.tasks:
            print(f"- {task['description']} (Role: {task['role']}, Priority: {task['priority']}, ID: {task['id']}, Dependencies: {task['dependencies']})")
        
        if not swarm.tasks:
            logger.info("All tasks completed!")
            break

    # Display created files
    files = swarm.file_ops.list_files()
    if files:
        logger.info("Created files:")
        for file in files:
            logger.info(f"- {file}")
    else:
        logger.info("No files were created during this run.")