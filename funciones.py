import json
from datetime import datetime
import pandas as pd
import re


from datetime import datetime
import pandas as pd

def transform_item(item):

    # Mapear status
    item["status_active"] = 1 if item.get("status") == "active" else 0
    item.pop("status", None)

    # One-hot mode
    mode = item.get("mode")
    if mode:
        item[f"mode_{mode}"] = 1
    item.pop("mode", None)

    # One-hot listing_type_id
    listing = item.get("listing_type_id", "").lower()
    for cat in ["gold", "silver", "bronze", "free"]:
        item[f"listing_{cat}"] = int(cat in listing)
    item.pop("listing_type_id", None)

    return item


