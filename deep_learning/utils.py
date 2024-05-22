import regex as re

def get_file_dir(source_path: str) -> str:
    """Retrieve directory of source path file"""

    pattern = r'(\S+/{1})\S+'

    # Use re.search to find the pattern in the file path
    match = re.search(pattern, source_path)
    
    # If a match is found, return the matched file extension
    if match:
        return match.group(1)
    else:
        return None  # Return None if no extension is found