from django.urls import path

from .views import SearchAPIView, SearchPageView

app_name = 'search'

urlpatterns = [
    path('', SearchPageView.as_view(), name='search-page'),
    path('api/search/', SearchAPIView.as_view(), name='api-search'),
]

