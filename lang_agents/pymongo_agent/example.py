# Example script demonstrating how to use the MongoDB Agent.
# This script allows users to interact with MongoDB using natural language.

import argparse
import pprint
import json

from langchain_core.messages import HumanMessage

from lang_agents.pymongo_agent.pymongo_agent import graph as mongodb_agent


MAX_NUMBER_OF_MESSAGES = 10


def print_response(response):
    try:
        json_response = json.loads(response)
        pprint.pprint(json_response)
    except json.JSONDecodeError:
        pprint.pprint(response)

def main():
    parser = argparse.ArgumentParser(description="MongoDB Agent - Query MongoDB using natural language")
    parser.add_argument("query", nargs="?", help="Natural language query for MongoDB")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        print("MongoDB Agent Interactive Mode")
        print("Enter your queries in natural language.")
        print("-" * 50)
        messages = []
        while True:
            try:
                user_input = input("\nEnter your query: ")
                
                if not user_input.strip():
                    continue
                
                message = HumanMessage(content=user_input)
                messages.append(message)
                result = mongodb_agent.invoke({"messages": messages})
                
                # Print the AI's response
                print_response(result["messages"][-1].content)

                messages = result["messages"]
                if len(messages) > MAX_NUMBER_OF_MESSAGES:
                    # Cut old messages to keep the conversation size manageable
                    messages = messages[-MAX_NUMBER_OF_MESSAGES:]
                
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt received. Exiting MongoDB Agent.")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    elif args.query:
        message = HumanMessage(content=args.query)
        result = mongodb_agent.invoke({"messages": [message]})
        print_response(result["messages"][-1].content)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 