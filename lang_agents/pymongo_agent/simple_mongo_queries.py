#!/usr/bin/env python3
# simple_mongo_queries.py - Simple MongoDB find and count queries

import os
from pymongo import MongoClient
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Get MongoDB connection string from environment variable
    mongo_uri = os.getenv("PYMONGO_HOST", "mongodb://localhost:27017/")
    
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    
    # Get database
    db = client['user_management']
    
    # Get collection
    users = db['users']
    
    print("=== Simple MongoDB Queries ===\n")
    
    # 1. Basic Find Query - Find all documents
    print("1. All users:")
    cursor = users.find()
    for doc in cursor:
        pprint(doc)
    print(f"Total documents: {users.count_documents({})}\n")
    
    # 2. Find with Filter - Find users in a specific state
    state = "CA"
    print(f"2. Users in {state}:")
    cursor = users.find({"address.state": state})
    for doc in cursor:
        pprint(doc)
    print(f"Count: {users.count_documents({'address.state': state})}\n")
    
    # 3. Find with Multiple Filters - Find users with specific theme and age range
    print("3. Users with dark theme and age between 25 and 40:")
    query = {
        "preferences.theme": "dark",
        "age": {"$gte": 25, "$lte": 40}
    }
    cursor = users.find(query)
    for doc in cursor:
        pprint(doc)
    print(f"Count: {users.count_documents(query)}\n")
    
    # 4. Find with Projection - Show only name and email
    print("4. All users (name and email only):")
    cursor = users.find({}, {"name": 1, "email": 1, "_id": 0})
    for doc in cursor:
        pprint(doc)
    print(f"Count: {users.count_documents({})}\n")
    
    # 5. Find with Sort - Sort by age descending
    print("5. Users sorted by age (descending):")
    cursor = users.find().sort("age", -1)
    for doc in cursor:
        print(f"{doc['name']} - Age: {doc.get('age', 'N/A')}")
    print(f"Count: {users.count_documents({})}\n")
    
    # 6. Count with Filter - Count users with notifications enabled
    query = {"preferences.notifications": True}
    count = users.count_documents(query)
    print(f"6. Users with notifications enabled: {count}\n")
    
    # 7. Find with Regex - Find users with email containing 'example'
    print("7. Users with email containing 'example':")
    cursor = users.find({"email": {"$regex": "example"}})
    for doc in cursor:
        print(f"{doc['name']} - {doc['email']}")
    print(f"Count: {users.count_documents({'email': {'$regex': 'example'}})}\n")
    
    # 8. Distinct Values - Get distinct states
    states = users.distinct("address.state")
    print(f"8. Distinct states: {states}\n")
    
    # 9. Find One - Get a single document
    print("9. First user:")
    doc = users.find_one()
    pprint(doc)
    print()
    
    # 10. Count by Group - Count users by state
    print("10. Count of users by state:")
    pipeline = [
        {"$group": {"_id": "$address.state", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    results = list(users.aggregate(pipeline))
    for result in results:
        print(f"State: {result['_id']}, Count: {result['count']}")

if __name__ == "__main__":
    main()
