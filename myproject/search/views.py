import json
import sqlite3
from pathlib import Path

import numpy as np
from django.conf import settings
from django.views.generic import TemplateView
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from sentence_transformers import SentenceTransformer

from .models import Item, ItemDetails
from .serializers import ItemSerializer


class SearchPageView(TemplateView):
    template_name = 'search/search.html'


class SearchAPIView(APIView):
    permission_classes = [AllowAny]
    _model = None
    _embeddings_cache = None
    _items_cache = None
    MODEL_NAME = "all-MiniLM-L6-v2"

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer(cls.MODEL_NAME)
        return cls._model

    @classmethod
    def _load_embeddings(cls):
        if cls._embeddings_cache is None:
            db_path = Path(settings.BASE_DIR) / "db.sqlite3"
            conn = sqlite3.connect(db_path)
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT item_id, name, category, embedding_json
                    FROM search_embedding
                    ORDER BY item_id;
                    """
                )
                rows = cur.fetchall()
            finally:
                conn.close()

            items = []
            vectors = []
            for row in rows:
                item_id, name, category, emb_json = row
                items.append((item_id, name, category))
                vectors.append(np.array(json.loads(emb_json), dtype=np.float32))

            cls._items_cache = items
            cls._embeddings_cache = np.vstack(vectors) if vectors else None
        return cls._items_cache, cls._embeddings_cache

    def post(self, request, *args, **kwargs):
        query = (request.data.get('query') or '').strip()
        if not query:
            return Response({'results': [], 'match_count': 0, 'message': 'Type something to search.'})

        items, embeddings = self._load_embeddings()
        if embeddings is None:
            return Response({'results': [], 'match_count': 0, 'message': 'No embeddings found. Run db_setup.py first.'})

        model = self._load_model()
        query_vec = model.encode(query, convert_to_numpy=True)
        similarities = embeddings @ query_vec

        indexed = list(zip(items, similarities))
        indexed.sort(key=lambda x: x[1], reverse=True)
        indexed = indexed[:2]

        item_ids = [item_id for (item_id, _, _), _ in indexed]
        items_dict = {item.id: item for item in Item.objects.filter(id__in=item_ids)}
        
        item_names = [item.name for item in items_dict.values()]
        details_dict = {details.name: details for details in ItemDetails.objects.filter(name__in=item_names)}

        results = []
        for (item_id, _, _), sim in indexed:
            if item_id in items_dict:
                item = items_dict[item_id]
                item.item_details = details_dict.get(item.name)
                serializer = ItemSerializer(item)
                result = serializer.data
                result['similarity'] = float(sim)
                result['distance'] = float(1.0 - sim)
                results.append(result)

        return Response({'results': results, 'match_count': len(results)})

