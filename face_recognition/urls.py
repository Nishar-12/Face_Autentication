from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('recognize/', views.recognize, name='recognize'),
    path('success/', views.success, name='success'),
]
