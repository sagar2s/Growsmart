from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('recommend', views.crop_recommend, name='recommend'),

    path('recomdata', views.crop_predict, name='recomdata'),
    path('coming',views.geo_coming, name='coming'),
    path('detect', views.disease_detect, name='detect'),
    path('fertstore', views.fertstore, name='fertstore'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)