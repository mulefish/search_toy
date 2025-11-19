from django.db import models


class Item(models.Model):
    name = models.CharField(max_length=120, unique=True)
    description = models.TextField()
    category = models.CharField(max_length=80)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']
        db_table = 'search_item'

    def __str__(self) -> str:
        return self.name


class ItemDetails(models.Model):
    name = models.CharField(max_length=120)
    number = models.FloatField()
    json_string = models.TextField()

    class Meta:
        db_table = 'search_item_details'

    def __str__(self) -> str:
        return f"{self.name} ({self.number})"

