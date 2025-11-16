from django.db.models import Q
from django.views.generic import TemplateView
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Item
from .serializers import ItemSerializer


class SearchPageView(TemplateView):
    template_name = 'search/search.html'


class SearchAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        query = (request.data.get('query') or '').strip()
        print("query: ", query)
        if not query:
            return Response({'results': [], 'match_count': 0, 'message': 'Type something to search.'})

        filters = Q()
        tokens = {query, *query.split()}
        for token in tokens:
            print("token: ", token)
            filters |= Q(name__icontains=token) | Q(description__icontains=token) | Q(category__icontains=token)

        qs = Item.objects.filter(filters).order_by('name')
        serializer = ItemSerializer(qs, many=True)
        return Response({'results': serializer.data, 'match_count': len(serializer.data)})

