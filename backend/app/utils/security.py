import random
import string

def generate_join_code(length: int = 5) -> str:
    """Generate a random join code."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

