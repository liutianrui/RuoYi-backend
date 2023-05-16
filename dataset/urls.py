from django.urls import path
from rest_framework.routers import SimpleRouter

from .views import CrudDemoModelViewSet

router = SimpleRouter()
router.register("api/dataset", CrudDemoModelViewSet)

urlpatterns = [
    path('api/dataset/Download/', CrudDemoModelViewSet.as_view({'get': 'Download'})),
    path('api/dataset/trainDataSet/', CrudDemoModelViewSet.as_view({'get': 'trainDataSet'})),
    path('api/dataset/model_download/', CrudDemoModelViewSet.as_view({'get': 'model_download'})),
]
urlpatterns += router.urls

