import asyncio
from typing import List, Dict, Any, TYPE_CHECKING

from concurrency.llm_core import llm_core
from tools.rag_utils import RAG, store_information
from tools.context_manager import context_manager

if TYPE_CHECKING:
    from agents.agent_init import Agent
    from ensemble.swarm import Swarm

class ChatEnvironment:
    def __init__(self, swarm: 'Swarm'):
        self.swarm = swarm
        self.chat_history: List[Dict[str, Any]] = []
        self.rag = RAG()

    async def start_chat(self):
        print("Welcome to the Agent Chat Environment!")
        print("Agents can communicate here and collaborate on tasks.")
        print("Type 'exit' to leave the chat environment.")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break

            await self.process_user_input(user_input)

    async def process_user_input(self, user_input: str):
        if user_input.startswith("@"):
            # Direct message to a specific agent
            parts = user_input.split(" ", 1)
            if len(parts) == 2:
                agent_name = parts[0][1:]
                message = parts[1]
                await self.send_message_to_agent(agent_name, message)
            else:
                print("Invalid format. Use @AgentName Message")
        else:
            # Broadcast message to all agents
            await self.broadcast_message("User", user_input)

    async def send_message_to_agent(self, agent_name: str, message: str):
        agent = self.swarm.get_agent_by_name(agent_name)
        if agent:
            response = await agent.ask_question(message)
            self.chat_history.append({"sender": "User", "receiver": agent_name, "message": message})
            self.chat_history.append({"sender": agent_name, "receiver": "User", "message": response})
            print(f"{agent_name}: {response}")
        else:
            print(f"Agent {agent_name} not found.")

    async def broadcast_message(self, sender: str, message: str):
        self.chat_history.append({"sender": sender, "receiver": "All", "message": message})
        print(f"{sender} (to all): {message}")
        
        responses = await asyncio.gather(*[agent.ask_question(message) for agent in self.swarm.agents if agent.activated])
        
        for agent, response in zip(self.swarm.agents, responses):
            if agent.activated:
                self.chat_history.append({"sender": agent.name, "receiver": "All", "message": response})
                print(f"{agent.name}: {response}")

    async def agent_message(self, sender: 'Agent', message: str, receiver: 'Agent' = None):
        if receiver:
            self.chat_history.append({"sender": sender.name, "receiver": receiver.name, "message": message})
            await self.store_message_in_rag(sender.name, receiver.name, message)
            await context_manager.add_to_context({"sender": sender.name, "receiver": receiver.name, "message": message})
            await receiver.process_incoming_message(sender, message)
        else:
            await self.broadcast_message(sender.name, message)

    async def store_message_in_rag(self, sender: str, receiver: str, message: str):
        context = f"Message from {sender} to {receiver}: {message}"
        await store_information(self.rag, context)

    async def get_relevant_context(self, query: str) -> str:
        return await llm_core.get_relevant_context(query, self.swarm.project_overview, self.chat_history)

    def display_chat_history(self):
        for entry in self.chat_history:
            if entry["receiver"] == "All":
                print(f"{entry['sender']} (to all): {entry['message']}")
            else:
                print(f"{entry['sender']} to {entry['receiver']}: {entry['message']}")

async def initialize_chat_environment(swarm: 'Swarm') -> ChatEnvironment:
    chat_env = ChatEnvironment(swarm)
    return chat_env