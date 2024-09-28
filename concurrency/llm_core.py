import os
from openai import OpenAI
from typing import Dict, Any, List, Callable
import json
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import logging

load_dotenv()  # Load environment variables from .env file

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set logging level to ERROR to disable debug logs
# logging.getLogger("openai").setLevel(logging.ERROR)
# logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").disabled = True
logging.getLogger("httpx").disabled = True

class LLMCore:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found in environment variables")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        self.assistants = {}
        self.threads = {}
        self.tool_functions = {}

    async def create_assistant(self, name: str, instructions: str, tools: List[Dict[str, Any]]):
        assistant = self.openai_client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=tools,
            model="gpt-4-1106-preview"
        )
        self.assistants[name] = assistant
        return assistant.id

    async def create_thread(self):
        thread = self.openai_client.beta.threads.create()
        return thread.id

    async def generate_response(self, assistant_name: str, prompt: str, thread_id: str = None) -> str:
        if thread_id is None:
            thread_id = await self.create_thread()
        
        logging.info(f"Generating response for {assistant_name} with prompt: {prompt[:50]}...")
        
        message = self.openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=prompt
        )

        run = self.openai_client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistants[assistant_name].id
        )

        while True:
            run_status = self.openai_client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            elif run_status.status == 'failed':
                logging.error(f"Run failed: {run_status.last_error}")
                return f"Error: {run_status.last_error}"
            await asyncio.sleep(1)

        messages = self.openai_client.beta.threads.messages.list(thread_id=thread_id)
        response = messages.data[0].content[0].text.value
        logging.info(f"Response generated for {assistant_name}: {response[:50]}...")
        return response

    def register_tool_function(self, function_name: str, function: Callable):
        self.tool_functions[function_name] = function

    async def delete_assistant(self, assistant_name: str):
        if assistant_name in self.assistants:
            self.openai_client.beta.assistants.delete(self.assistants[assistant_name].id)
            del self.assistants[assistant_name]
            if assistant_name in self.threads:
                del self.threads[assistant_name]

    async def gemini_generate_content(self, prompt: str) -> str:
        response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
        return response.text

    async def get_relevant_context(self, query: str, project_overview: str, context: List[Dict[str, Any]]) -> str:
        prompt = f"""Project Overview: {project_overview}

Given the following query: "{query}"
And this context: {context}
Provide a concise summary of the most relevant information from the context that relates to the query.
Include only the most important details and limit the response to 2000 words."""

        return await self.gemini_generate_content(prompt)

    async def summarize_for_new_agent(self, agent_role: str, task_description: str, project_overview: str, context: List[Dict[str, Any]]) -> str:
        prompt = f"""Project Overview: {project_overview}

New Agent Role: {agent_role}
Task Description: {task_description}

Given the above information and the following context of previous agent interactions and tasks:
{context}

Provide a concise summary of the most relevant information that this new agent should know to perform its task effectively.
Include key points from the project overview, relevant previous agent interactions, and any crucial information related to the task.
Limit the response to 2000 words."""

        return await self.gemini_generate_content(prompt)

llm_core = LLMCore()