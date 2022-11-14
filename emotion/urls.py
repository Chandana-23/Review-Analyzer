from django.urls import path,include
from . import views

urlpatterns = [
    path('',views.home,name='/'),
    path('detect',views.detect,name='detect'),
]