
#%%
import json
import uuid
from datetime import datetime

# Generate timestamp and UUID
run_data = {
    "timestamp": datetime.now().strftime('%Y-%m-%d_H%H-M%M-S%S'),
    "uuid": str(uuid.uuid4())
}

# Save to run_parameters file
with open(r'utils\run_parameters.json', 'w') as f:
    json.dump(run_data, f)


# %%
