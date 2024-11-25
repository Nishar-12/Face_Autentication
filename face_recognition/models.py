from django.db import models

# Create your models here.
from django.db import models

class User(models.Model):
    username = models.CharField(max_length=50, unique=True)
    image = models.ImageField(upload_to='faces/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username
