from django.urls import path
from . import views

urlpatterns = [
    path('',                          views.home,            name='home'),
    path('assessments/',              views.property_list,   name='property_list'),
    path('assessments/new/',          views.property_create, name='property_create'),
    path('assessments/<int:pk>/',     views.property_detail, name='property_detail'),
    path('assessments/<int:pk>/edit/',views.property_update, name='property_update'),
    path('assessments/<int:pk>/delete/', views.property_delete, name='property_delete'),
    path('assessments/<int:pk>/ai-explain/', views.ai_explain, name='ai_explain'),
]
