#!/usr/bin/env python
"""
Example script demonstrating how to use the MongoDB Agent.
This script allows users to interact with MongoDB using natural language.
"""

import sys
import argparse
import pprint

from langchain_core.messages import HumanMessage

from lang_agents.pymongo_agent.pymongo_agent import graph as mongodb_agent

def main():
    parser = argparse.ArgumentParser(description="MongoDB Agent - Query MongoDB using natural language")
    parser.add_argument("query", nargs="?", help="Natural language query for MongoDB")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        print("MongoDB Agent Interactive Mode")
        print("Type 'exit' or 'quit' to exit")
        print("Enter your queries in natural language.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nEnter your query: ")
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting MongoDB Agent. Goodbye!")
                    break
                
                if not user_input.strip():
                    continue
                
                message = HumanMessage(content=user_input)
                result = mongodb_agent.invoke({"messages": [message]})
                
                # Print the AI's response
                print("\n" + result.messages[-1].content)
                print("\n" + "-" * 50)
                
            except KeyboardInterrupt:
                print("\nExiting MongoDB Agent. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    elif args.query:
        message = HumanMessage(content=args.query)
        result = mongodb_agent.invoke({"messages": [message]})
        pprint.pprint(result["messages"])
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 