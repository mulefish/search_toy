from rest_framework import serializers

from .models import Item, ItemDetails


class ItemDetailsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ItemDetails
        fields = ('number', 'json_string')


class ItemSerializer(serializers.ModelSerializer):
    details = ItemDetailsSerializer(read_only=True, source='item_details')

    class Meta:
        model = Item
        fields = ('id', 'name', 'description', 'category', 'details')

