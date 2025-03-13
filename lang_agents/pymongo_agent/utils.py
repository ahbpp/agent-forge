import os
import pymongo




def get_read_mongo_client():
    return pymongo.MongoClient(
        host=os.getenv("PYMONGO_HOST"),
        username=os.getenv("PYMONGO_USER"),
        password=os.getenv("PYMONGO_PASSWORD"),
        authSource="admin",
        read_preference=pymongo.ReadPreference.SECONDARY
    )


def print_schema(collection):
    """
    Print the schema of a collection for each field. It can be several levels deep.
    in format field: type
    """
    schema = collection.find_one()
    def print_schema_recursive(schema, prefix=""):
        if isinstance(schema, dict):
            for key, value in schema.items():
                print_schema_recursive(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(schema, list):
            print_schema_recursive(schema[0], prefix)
        else:
            print(f"{prefix}: {type(schema)}")

    print_schema_recursive(schema)
