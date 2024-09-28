import os
import asyncio
from typing import List, Dict, Any
from ensemble.swarmify import initialize_swarm, run_swarm
from tools.file_operations import FileOperations
import logging

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("openai").disabled = True
logging.getLogger("httpx").disabled = True

async def main():
    project_goal = input("Enter the project goal: ")
    project_overview = input("Enter a brief project overview: ")
    
    try:
        swarm = await initialize_swarm(project_goal, project_overview)
    except Exception as e:
        logging.error(f"Error initializing swarm: {str(e)}")
        return

    print("\nCreated agents:")
    for agent in swarm.agents:
        print(f"- {agent.name} ({agent.role})")
    
    print("\nInitial tasks:")
    for task in swarm.tasks:
        print(f"- {task['description']} (Role: {task['role']}, Priority: {task['priority']}, ID: {task['id']}, Dependencies: {task['dependencies']})")

    file_ops = FileOperations()

    iteration_count = 0
    max_iterations = 10  # Set a maximum number of iterations to prevent endless loops

    while iteration_count < max_iterations:
        iteration_count += 1
        logging.info(f"Starting iteration {iteration_count}")

        try:
            await run_swarm(swarm, iterations=1)
        except Exception as e:
            logging.error(f"Error during swarm iteration: {str(e)}")

        # Check for file creation
        files = file_ops.list_files()
        if files:
            logging.info("Files created in this iteration:")
            for file in files:
                logging.info(f"- {file}")
                content = file_ops.read_file(file)
                logging.info(f"  Content preview: {content[:100]}...")
        else:
            logging.info("No files were created in this iteration.")

        # Check if all tasks are completed
        if not swarm.tasks:
            logging.info("All tasks completed!")
            break

        print("\nCurrent tasks:")
        for task in swarm.tasks:
            print(f"- {task['description']} (Role: {task['role']}, Priority: {task['priority']}, ID: {task['id']}, Dependencies: {task['dependencies']})")
        print("\nCompleted tasks:")
        for task in swarm.completed_tasks:
            print(f"- {task['description']} (Role: {task['role']}, ID: {task['id']})")

        user_input = input("\nPress Enter to continue to the next iteration, or 'q' to quit: ")
        if user_input.lower() == 'q':
            break

    if iteration_count == max_iterations:
        logging.warning(f"Reached maximum number of iterations ({max_iterations}) without completing all tasks.")

    print("\nFinal state:")
    print("Remaining tasks:")
    for task in swarm.tasks:
        print(f"- {task['description']} (Role: {task['role']}, Priority: {task['priority']}, ID: {task['id']}, Dependencies: {task['dependencies']})")
    print("\nCompleted tasks:")
    for task in swarm.completed_tasks:
        print(f"- {task['description']} (Role: {task['role']}, ID: {task['id']})")

    files = file_ops.list_files()
    if files:
        print("\nCreated files:")
        for file in files:
            print(f"- {file}")
            content = file_ops.read_file(file)
            print(f"  Content preview: {content[:100]}...")
    else:
        print("\nNo files were created during the entire run.")

if __name__ == "__main__":
    asyncio.run(main())