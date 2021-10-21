from django.db import models

# Create your models here.

class DatasetModel(models.Model):
    name = models.CharField(max_length=255, default = '')
    categories = models.CharField(max_length=5000, default = '',blank=True)
    images_path = models.CharField(max_length=255, default = '')
    def __str__(self):
        return str(self.name)

class ImageModel(models.Model):
    id = models.AutoField(primary_key=True)
    filename = models.CharField(max_length=255, default = '')
    img_url = models.CharField(max_length=500, default = '')
    europeana_id = models.CharField(max_length=500, default = '')
    uri = models.CharField(max_length=500, default = '')
    dataset = models.ManyToManyField(DatasetModel)
    def __str__(self):
        return str(self.filename)
