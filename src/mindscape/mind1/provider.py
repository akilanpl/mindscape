"""External inference removed from the product. PR27 retains its archived implementation."""
EXTERNAL_INFERENCE_ENABLED = False

def status():
    return {"enabled": False, "reason": "Mind 1 uses only Mindscape-native inference and explicitly attributed tools"}
