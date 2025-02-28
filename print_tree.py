import os

def print_directory_tree(directory, indent="", prefix=""):
    """
    Recursively print the directory structure in a tree-like format.
    """
    # Get all items in the directory
    items = sorted(os.listdir(directory))
    # Filter out common ignored files/folders (like __pycache__, .git, etc.)
    items = [item for item in items if item not in {'.git', '__pycache__', '.vscode', '.gitignore'}]
    
    for index, item in enumerate(items):
        path = os.path.join(directory, item)
        is_last = index == len(items) - 1
        # Set the connector based on whether it's the last item
        connector = "└── " if is_last else "├── "
        
        # Print the current item
        print(f"{indent}{prefix}{connector}{item}")
        
        # If it's a directory, recurse into it
        if os.path.isdir(path):
            new_prefix = "    " if is_last else "│   "
            print_directory_tree(path, indent, new_prefix)

# Set the root directory to the current working directory (or specify your project path)
root_dir = os.getcwd()  # You can replace this with a specific path like "C:/path/to/trade_bot"
print(f"Project structure at: {root_dir}")
print_directory_tree(root_dir)