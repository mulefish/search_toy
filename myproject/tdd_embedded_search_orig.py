"""
Semantic Search Demo for Cannabis Product Descriptions
Uses sentence-transformers for embedding-based search
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

ITEMS = [
    (
        "Indica Reverie",
        "Indica drapes the evening in velvet calm, melting every muscle into the couch while citrus-lavender terpenes lull your mind toward cinematic dreams and whispery conversations that feel like bonus tracks from a favorite album. The buzz lingers like a weighted blanket, turning every flicker of candlelight into a widescreen spectacle and inviting deep breaths of hush.",
        "Indica",
    ),
    (
        "Sativa Voltage",
        "Sativa fires up neon-focus swagger, a sunrise surge that sends your thoughts sprinting across skyline billboards with laser-sharp wit and unstoppable creative flow. Each inhale feels like cracking open a can of inspiration, launching brainstorming sessions, rooftop hangouts, and fearless to-do list takedowns with equal amounts of gleam.",
        "Sativa",
    ),
    (
        "Hybrid Flux",
        "Hybrid splices the genetics of hustle and hush, letting you toggle between brainstorm brilliance and after-hours bliss with a single smooth inhale. It's the Swiss Army buzz—boardroom presentable one minute, fire-pit stories the next—keeping your vibe cruising on adaptive cruise control no matter how the day rearranges itself.",
        "Hybrid",
    ),
    (
        "Wax Prism",
        "Wax condenses flavor galaxies into molten gold, snapping onto your rig with terpene fireworks that crackle like a headline festival drop. One pull paints your palate with citrus candy, pine forest, and backstage-pass confidence, leaving a shimmering trail of dense clouds that shout premium without saying a word.",
        "Wax",
    ),
    (
        "Pre-Roll Ember",
        "Pre-rolls roll out concierge convenience, artisan-packed cones that spark evenly and unfurl luxury smoke trails wherever the night takes you. Slide a tube into your pocket and you're packing a mobile lounge—perfectly ground flower, slow-burning papers, and a vibe that says you paid attention to the finer details.",
        "Pre-roll",
    ),
    (
        "Gummy Aurora",
        "Gummys glow with technicolor delight, micro-dosed jewels that melt beneath the tongue and glide you through a syrupy spectrum of euphoria. Each bite is a postcard from Candyland—precision-dosed, terpene-kissed, and ready to turn playlists, painting sessions, or spa-night rituals into a silky-smooth adventure.",
        "Gummys",
    ),
    (
        "Pre-Rolls Duo",
        "Pre-rolls pair up in twin packs for co-op adventures, sharing boutique flower that burns clean, smells like a terpene boutique, and keeps the vibe synchronized from first spark to final ash. Whether you're matching cones with a friend or saving one for later, each wrap feels like a pocket-sized house party.",
        "Pre-rolls",
    ),
]


class SemanticProductSearch:
    """In-memory semantic search using embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a sentence transformer model.
        
        Models to consider:
        - 'all-MiniLM-L6-v2': Fast, good quality (default)
        - 'all-mpnet-base-v2': Higher quality, slower
        - 'multi-qa-mpnet-base-dot-v1': Optimized for Q&A/search
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.items = []
        self.embeddings = None
        
    def index_products(self, products: List[Tuple[str, str, str]]):
        """
        Index products by creating embeddings for name + description.
        
        Args:
            products: List of (name, description, category) tuples
        """
        self.items = products
        
        # Combine name and description for richer embeddings
        texts = [f"{name}. {description}" for name, description, _ in products]
        
        print(f"Encoding {len(texts)} products...")
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        print(f"Embeddings shape: {self.embeddings.shape}")
        
    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, str, str, str]]:
        """
        Search for products matching the query semantically.
        
        Args:
            query: Search query (natural language)
            top_k: Number of results to return
            
        Returns:
            List of (score, name, description, category) tuples, sorted by relevance
        """
        if self.embeddings is None:
            raise ValueError("No products indexed. Call index_products() first.")
        
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Compute cosine similarity
        # Using dot product since embeddings are normalized
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            name, description, category = self.items[idx]
            results.append((score, name, description, category))
        
        return results
    
    def print_results(self, query: str, results: List[Tuple[float, str, str, str]]):
        """Pretty print search results"""
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}\n")
        
        for i, (score, name, description, category) in enumerate(results, 1):
            print(f"{i}. {name} ({category}) - Score: {score:.4f}")
            print(f"   {description[:150]}...")
            print()


# Example usage and demo
if __name__ == "__main__":
    # Initialize search engine
    search_engine = SemanticProductSearch(model_name='all-MiniLM-L6-v2')
    
    # Index all products
    search_engine.index_products(ITEMS)
    
    # Demo queries showing semantic understanding
    test_queries = [
        "something to help me relax after work",
        "I need energy and focus for creative work",
        "balanced option for any time of day",
        "concentrate with strong flavor",
        "convenient option I can take anywhere",
        "edible that tastes good",
        "citrus flavors",
        "help me sleep",
        "brainstorming session",
        "couch lock",
    ]
    
    print("\n" + "="*80)
    print("SEMANTIC SEARCH DEMO")
    print("="*80)
    
    for query in test_queries:
        results = search_engine.search(query, top_k=3)
        search_engine.print_results(query, results)
        input("Press Enter for next query...")


# For integration into Django/your app:
"""
class Product(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    category = models.CharField(max_length=100)
    # Add this field for pgvector
    # embedding = VectorField(dimensions=384)  # for all-MiniLM-L6-v2
    
# In your view:
def search_products(request):
    query = request.GET.get('q', '')
    
    if query:
        # Option 1: Pure in-memory (for small datasets)
        search_engine = SemanticProductSearch()
        products = Product.objects.all().values_list('name', 'description', 'category')
        search_engine.index_products(list(products))
        results = search_engine.search(query)
        
        # Option 2: With pgvector (for larger datasets)
        # query_embedding = model.encode(query)
        # results = Product.objects.order_by(
        #     RawSQL("embedding <-> %s", [query_embedding])
        # )[:10]
    
    return render(request, 'search.html', {'results': results})
"""