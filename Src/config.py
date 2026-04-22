from openai import OpenAI

# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI("Add Key Here")
# ============================================================
# FILE PATHS
# ============================================================

DESC_INDEX_FILE = "spend_desc_index.faiss"
GL_INDEX_FILE = "spend_gl_index.faiss"
METADATA_FILE = "metadata.pkl"
SUPPLIER_MAP_FILE = "supplier_map.pkl"

# ============================================================
# SEARCH SETTINGS
# ============================================================

TOP_K_SEARCH = 20
MIN_K = 3
MAX_K = 7

SIMILARITY_THRESHOLD = 0.65

AUTO_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.70