"""Seed the memory bank with high-quality starter strategies."""
import uuid
from datetime import datetime
from src.models import MemoryItem
from src.memory import ReasoningBank

# High-quality seed memories based on successful WebArena patterns
SEED_MEMORIES = [
    {
        "title": "Using Map Directions for Distance Calculation",
        "description": "To find driving distance between two locations, use the map's 'Directions' feature with explicit From/To fields",
        "content": [
            "Navigate to the map service and use the search to find both locations first.",
            "Click on 'Directions' button to access the routing interface.",
            "Enter the starting location in the 'From' field and destination in the 'To' field.",
            "The route will display distance and driving time directly.",
            "For national parks or landmarks, search by specific name rather than generic terms."
        ]
    },
    {
        "title": "Specific Location Search Before Generic Terms",
        "description": "When searching for a specific type of location (park, bridge, building), identify the exact name first before attempting map search",
        "content": [
            "If the task asks for 'closest national park', first identify which specific park it is (e.g., Acadia, Yellowstone).",
            "Use Wikipedia or knowledge base to find the specific name.",
            "Then search the map using that specific name, not generic terms like 'national park'.",
            "Map search engines work better with proper nouns than with categories."
        ]
    },
    {
        "title": "Form Navigation and Element Identification",
        "description": "When interacting with forms and dropdown menus, understand the element hierarchy",
        "content": [
            "For dropdown menus, click on the [combobox] parent element, NOT the [option] children.",
            "If you see [combobox] (id: 23) with [option] children, click ID 23 to open the dropdown.",
            "For search boxes, use type() on [textbox] elements, then look for a search or submit button.",
            "Read the page structure before acting - identify form elements, their types, and relationships."
        ]
    },
    {
        "title": "Reddit Forum Navigation - Direct URL Approach",
        "description": "Navigate directly to Reddit forums by URL instead of searching or scrolling through the forums list",
        "content": [
            "Reddit forum URLs follow pattern: /f/[forum_name] (e.g., /f/DIY, /f/worcester, /f/showerthoughts)",
            "ALWAYS try direct navigation FIRST: navigate('http://ec2.../f/forum_name')",
            "Forum names in URLs are case-sensitive and use lowercase (Worcester → worcester)",
            "If direct navigation fails (404), THEN try the alphabetical list at /forums/all",
            "Avoid scrolling through long forum lists - limit to 3-4 scroll attempts max",
            "The search box on /forums page often doesn't filter results - don't rely on it"
        ]
    },
    {
        "title": "GitLab Project Discovery via Projects Page",
        "description": "To find GitLab projects, use the Projects menu rather than search functionality",
        "content": [
            "Navigate to Projects > Your Projects or Projects > Explore to see project listings.",
            "Projects are typically listed with clear titles and metadata.",
            "Click on project name to access the repository.",
            "For project file contents (like README), navigate inside the project to the file listing."
        ]
    },
    {
        "title": "Wikipedia Article Navigation by Categories",
        "description": "If Wikipedia search fails, use the category browse system or main page navigation",
        "content": [
            "Access the Main Page to find featured articles and category links.",
            "Look for category navigation (e.g., 'People', 'Places', 'Science').",
            "Navigate through category hierarchies to find specific topics.",
            "Once on a related article, use internal links to navigate to the target article."
        ]
    },
    {
        "title": "E-commerce Product Search and Filtering",
        "description": "To find products in specific price ranges or with certain features, use the shop's filter system",
        "content": [
            "Start with a category search or general product search.",
            "Use filter controls on the left sidebar for price range, rating, brand, etc.",
            "For price ranges, input min and max values in the price filter fields.",
            "Apply filters one at a time and verify results update before adding more filters."
        ]
    },
    {
        "title": "Extracting Structured Data from Product Listings",
        "description": "When collecting product information (titles, prices, reviews), systematically scan the product listing page",
        "content": [
            "Product listings typically show: title, price, rating, and review count.",
            "For 'top N products', count from the top of the search results.",
            "Extract exact text - don't paraphrase product names or prices.",
            "For price ranges, format as '$min - $max' with space around the dash."
        ]
    },
]

def create_seed_memory(mem_dict: dict) -> MemoryItem:
    """Create a MemoryItem from a dictionary."""
    return MemoryItem(
        id=str(uuid.uuid4()),
        title=mem_dict["title"],
        description=mem_dict["description"],
        content=mem_dict["content"],
        rationale=None,  # Optional for seed memories
        provenance={
            "task_id": "seed",
            "success": True,  # Mark as successful strategies
            "timestamp": datetime.now().isoformat(),
            "steps": 0,
            "source": "manual_seed"
        },
        embedding=None  # Will be generated during add
    )

def seed_memory_bank(bank_path: str = "memory_bank"):
    """Seed the memory bank with high-quality strategies."""
    print("Initializing embedding provider...")
    from src.embeddings import create_embedding_provider
    import os
    
    # Priority: Google Gemini > OpenAI > sentence transformers > simple hash
    if os.getenv("GOOGLE_API_KEY"):
        embedding_provider = create_embedding_provider("google", "models/embedding-001")
        print("  Using Google Gemini embeddings")
    elif os.getenv("OPENAI_API_KEY"):
        embedding_provider = create_embedding_provider("openai", "text-embedding-3-large")
        print("  Using OpenAI embeddings")
    else:
        try:
            embedding_provider = create_embedding_provider("sentence_transformers", "sentence-transformers/all-MiniLM-L6-v2")
            print("  Using SentenceTransformer embeddings")
        except Exception:
            embedding_provider = create_embedding_provider("simple", "")
            print("  Using simple hash embeddings")
    
    print(f"Loading memory bank from {bank_path}...")
    bank = ReasoningBank(
        bank_path=bank_path,
        embedding_provider=embedding_provider,
        dedup_threshold=0.9
    )
    
    print(f"Current memory bank has {len(bank.memories)} memories")
    print(f"\nAdding {len(SEED_MEMORIES)} seed memories...")
    
    added_count = 0
    for mem_dict in SEED_MEMORIES:
        memory = create_seed_memory(mem_dict)
        if bank.add_memory(memory, check_duplicate=True):
            added_count += 1
            print(f"  ✓ Added: {memory.title}")
        else:
            print(f"  ⊘ Skipped (duplicate): {memory.title}")
    
    print(f"\n✓ Successfully added {added_count} new memories")
    print(f"Total memories in bank: {len(bank.memories)}")
    
    # Save the index
    bank.save_checkpoint()
    print(f"✓ Saved memory bank to {bank_path}")

if __name__ == "__main__":
    seed_memory_bank()

