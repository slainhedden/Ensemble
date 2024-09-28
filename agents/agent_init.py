from typing import List, Dict, Any, Optional
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from interaction.chat_environment import ChatEnvironment
from tools.rag_utils import RAG, store_information, get_knowledge
from tools.web_search import WebSearch
from tools.file_operations import FileOperations
from concurrency.llm_core import llm_core
import asyncio
from tools.context_manager import context_manager
import logging

logger = logging.getLogger(__name__)
logging.getLogger("openai").disabled = True
logging.getLogger("httpx").disabled = True

class Agent:
    def __init__(self, name: str, role: str, specialties: List[str]):
        self.name = name
        self.role = role
        self.specialties = specialties
        self.assistant_id: Optional[str] = None
        self.thread_id: Optional[str] = None
        self.rag: Optional[RAG] = None
        self.chat_env: Optional['ChatEnvironment'] = None
        self.current_task: Optional[Dict[str, Any]] = None
        self.file_ops = FileOperations()
        self.web_search = WebSearch()
        self.completed_tasks: List[Dict[str, Any]] = []
        self.activated = False

    async def initialize_with_context(self, task_description: str, project_overview: str):
        instructions = f"""You are {self.name}, a {self.role} with expertise in {', '.join(self.specialties)}.
Your task is: {task_description}
Project overview: {project_overview}

You have access to the following tools:
1. get_knowledge: Retrieve relevant information from the knowledge base.
2. web_search_and_learn: Perform a web search and learn from the results.
3. read_file: Read the content of a file.
4. write_file: Write content to a file.
5. append_file: Append content to an existing file.
6. store_information: Store important information in the knowledge base.

Use these tools to complete your tasks efficiently. Always consider the project overview and your specific role when making decisions."""

        self.assistant_id = await llm_core.create_assistant(
            self.name,
            instructions,
            [
                {"type": "function", "function": {"name": "get_knowledge", "description": "Retrieve relevant information from the knowledge base"}},
                {"type": "function", "function": {"name": "web_search_and_learn", "description": "Perform a web search and learn from the results"}},
                {"type": "function", "function": {"name": "read_file", "description": "Read the content of a file"}},
                {"type": "function", "function": {"name": "write_file", "description": "Write content to a file"}},
                {"type": "function", "function": {"name": "append_file", "description": "Append content to an existing file"}},
                {"type": "function", "function": {"name": "store_information", "description": "Store important information in the knowledge base"}},
            ]
        )
        self.thread_id = await llm_core.create_thread()
        logger.info(f"Initialized agent: {self.name} ({self.role})")

    async def execute_task(self, task: Dict[str, Any]) -> str:
        prompt = f"Execute the following task: {task['description']}\n\nProvide a detailed plan and then execute it step by step. Use the available tools when necessary."
        response = await llm_core.generate_response(self.name, prompt, self.thread_id)
        logger.info(f"{self.name} executed task: {task['description']}")
        self.completed_tasks.append(task)
        return response

    async def ask_question(self, question: str) -> str:
        response = await llm_core.generate_response(self.name, question, self.thread_id)
        logger.info(f"{self.name} asked question: {question}")
        return response

    async def process_incoming_message(self, sender: 'Agent', message: str):
        response = await self.ask_question(f"Respond to this message from {sender.name}: {message}")
        await self.send_message(response, sender)

    def connect_to_chat_environment(self, chat_env):
        self.chat_env = chat_env
        logger.info(f"{self.name} connected to chat environment")

    async def send_message(self, message: str, receiver: 'Agent'):
        if self.chat_env:
            await self.chat_env.agent_message(self, message, receiver)
        else:
            logger.warning(f"{self.name} tried to send a message, but is not connected to a chat environment.")

    def get_shareable_knowledge(self) -> str:
        return f"Shareable knowledge from {self.name}: {', '.join(self.specialties)}"

    def can_activate(self) -> bool:
        return True

    def activate(self):
        self.activated = True
        logger.info(f"{self.name} has been activated.")

# Register tool functions
llm_core.register_tool_function("get_knowledge", get_knowledge)
llm_core.register_tool_function("web_search_and_learn", WebSearch().search)
llm_core.register_tool_function("read_file", FileOperations().read_file)
llm_core.register_tool_function("write_file", FileOperations().write_file)
llm_core.register_tool_function("store_information", store_information)
llm_core.register_tool_function("append_file", FileOperations().append_file)