import sys
import os
from pathlib import Path

# Always run the real main.py from the correct location
main_path = Path(__file__).parent.parent / 'Breed_Rec' / 'src' / 'main.py'
if not main_path.exists():
    sys.stderr.write(f"Error: {main_path} does not exist.\n")
    sys.exit(1)

os.execv(sys.executable, [sys.executable, str(main_path)])
