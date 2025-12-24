from datetime import datetime

def handle():
    return f"Today is {datetime.now().strftime('%A %d %B')}"
