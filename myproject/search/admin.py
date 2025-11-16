from django.contrib import admin

from .models import Item


@admin.register(Item)
class ItemAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'updated_at')
    search_fields = ('name', 'description', 'category')
    list_filter = ('category',)

