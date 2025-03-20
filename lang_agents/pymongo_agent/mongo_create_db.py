#!/usr/bin/env python3
# mongo_create_db.py - Script to create a MongoDB database with a collection and sample data

import dateutil.parser

from pymongo import MongoClient
from pprint import pprint

from datetime import datetime
    

# Define the collection schema with validation (at least two levels of nesting)
collection_schema = {
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["name", "email", "address", "created_at"],
            "properties": {
                "name": {
                    "bsonType": "string",
                    "description": "Name must be a string and is required"
                },
                "email": {
                    "bsonType": "string",
                    "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
                    "description": "Email must be a valid email address and is required"
                },
                "address": {
                    "bsonType": "object",
                    "required": ["street", "city", "state", "zip"],
                    "properties": {
                        "street": {
                            "bsonType": "string",
                            "description": "Street must be a string and is required"
                        },
                        "city": {
                            "bsonType": "string",
                            "description": "City must be a string and is required"
                        },
                        "state": {
                            "bsonType": "string",
                            "description": "State must be a string and is required"
                        },
                        "zip": {
                            "bsonType": "string",
                            "description": "Zip code must be a string and is required"
                        }
                    }
                },
                "phone": {
                    "bsonType": "string",
                    "description": "Phone must be a string if provided"
                },
                "age": {
                    "bsonType": "int",
                    "minimum": 18,
                    "maximum": 120,
                    "description": "Age must be an integer between 18 and 120 if provided"
                },
                "created_at": {
                    "bsonType": "date",
                    "description": "Created date is required"
                },
                "preferences": {
                    "bsonType": "object",
                    "properties": {
                        "notifications": {
                            "bsonType": "bool",
                            "description": "Notifications preference must be a boolean if provided"
                        },
                        "theme": {
                            "bsonType": "string",
                            "enum": ["light", "dark", "system"],
                            "description": "Theme must be one of: light, dark, system if provided"
                        }
                    }
                }
            }
        }
    }
}

# Sample data to insert into the collection
sample_data = [
    {
        "name": "Mike Edwards",
        "email": "mike.edwards@example.com",
        "address": {
            "street": "645 5th Ave",
            "city": "New York",
            "state": "NY",
            "zip": "11001",
            "country": "USA"
        },
        "phone": "123-456-7890",
        "age": 32,
        "created_at": "2025-03-18T00:00:00Z",
        "preferences": {
            "notifications": False,
            "display_name": "Mike Edwards"
        }
    },
    {
        "name": "Rachel Patel",
        "email": "rachel.patel@example.com",
        "address": {
            "street": "1 Market St",
            "city": "San Francisco",
            "state": "CA",
            "zip": "94105",
            "country": "USA"
        },
        "phone": "415-123-4567",
        "age": 28,
        "created_at": "2025-03-19T00:00:00Z",
        "preferences": {
            "notifications": True,
            "display_name": "Rae"
        }
    },
    {
        "name": "Juan Sanchez",
        "email": "juan.sanchez@example.com",
        "address": {
            "street": "1200 S Figueroa St",
            "city": "Los Angeles",
            "state": "CA",
            "zip": "90015",
            "country": "USA"
        },
        "phone": "213-456-7890",
        "age": 41,
        "created_at": "2023-03-20T00:00:00Z",
        "preferences": {
            "notifications": False,
            "display_name": "Juan"
        }
    },
    {
        "name": "Emily Chen",
        "email": "emily.chen@example.com",
        "address": {
            "street": "333 Washington St",
            "city": "Buffalo",
            "state": "NY",
            "zip": "14203",
            "country": "USA"
        },
        "phone": "716-123-4567",
        "age": 35,
        "created_at": "2021-03-21T00:00:00Z",
        "preferences": {
            "notifications": True,
            "display_name": "Emily"
        }
    },
    {
        "name": "David Lee",
        "email": "david.lee@example.com",
        "address": {
            "street": "1600 Broadway",
            "city": "New York",
            "state": "NY",
            "zip": "10019",
            "country": "USA"
        },
        "phone": "212-456-7890",
        "age": 38,
        "created_at": "2022-03-22T00:00:00Z",
        "preferences": {
            "notifications": False,
            "display_name": "David"
        }
    },
    {
        "name": "Sarah Johnson",
        "email": "sarah.johnson@example.com",
        "address": {
            "street": "201 W Wisconsin Ave",
            "city": "Milwaukee",
            "state": "WI",
            "zip": "53203",
            "country": "USA"
        },
        "phone": "414-123-4567",
        "age": 42,
        "created_at": "2024-04-23T00:00:00Z",
        "preferences": {
            "notifications": True,
            "display_name": "Sarah"
        }
    },
    {
        "name": "Michael Brown",
        "email": "michael.brown@example.com",
        "address": {
            "street": "123 S 15th St",
            "city": "Philadelphia",
            "state": "PA",
            "zip": "19102",
            "country": "USA"
        },
        "phone": "215-123-4567",
        "age": 45,
        "created_at": "2024-03-24T00:00:00Z",
        "preferences": {
            "notifications": False,
            "display_name": "Mike"
        }
    },
    {
        "name": "Amit Chaudhary",
        "email": "amit.chaudhary@example.com",
        "address": {
            "street": "1000 Rue de la Gauchetiere O",
            "city": "Montreal",
            "state": "QC",
            "zip": "H3G 1A1",
            "country": "Canada"
        },
        "phone": "514-123-4567",
        "age": 36,
        "created_at": "2024-03-25T00:00:00Z",
        "preferences": {
            "notifications": False,
            "display_name": "Amit"
        }
    },
    {
        "name": "Samantha Lee",
        "email": "samantha.lee@example.com",
        "address": {
            "street": "300 Front St W",
            "city": "Toronto",
            "state": "ON",
            "zip": "M5V 0E9",
            "country": "Canada"
        },
        "phone": "416-123-4567",
        "age": 39,
        "created_at": "2024-03-26T00:00:00Z",
        "preferences": {
            "notifications": True,
            "display_name": "Sam"
        }
    },
]

def main():
    # Connect to MongoDB (assumes MongoDB is running on localhost:27017)
    client = MongoClient('mongodb://localhost:27017/')
    
    # Create or get the database
    db_name = 'user_management'
    db = client[db_name]
    
    # Check if database exists in list
    print(f"Available databases: {client.list_database_names()}")
    
    # Drop the collection if it exists (for demonstration purposes)
    if 'users' in db.list_collection_names():
        db.users.drop()
        print("Dropped existing 'users' collection")
    
    # Create a new collection with schema validation
    print(f"Creating 'users' collection with schema validation...")
    db.create_collection('users', **collection_schema)
    
    # Get the collection
    users = db.users
    
    # Convert string dates to datetime objects for MongoDB

    for user in sample_data:
        if isinstance(user['created_at'], str):
            user['created_at'] = dateutil.parser.parse(user['created_at'])
    
    # Insert the sample data
    result = users.insert_many(sample_data)
    print(f"Inserted {len(result.inserted_ids)} documents")
    
    # Verify the data was inserted
    print("\nVerifying inserted data:")
    for user in users.find():
        pprint(user)
    
    # Show collection validation rules
    print("\nCollection validation rules:")
    pprint(db.command('listCollections', filter={'name': 'users'})['cursor']['firstBatch'][0]['options']['validator'])
    
    print(f"\nDatabase '{db_name}' and collection 'users' successfully created with schema validation.")

if __name__ == "__main__":
    main()
