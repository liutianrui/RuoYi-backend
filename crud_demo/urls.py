from rest_framework.routers import SimpleRouter

from .views import DataSetModelViewSet
from django.urls import path

router = SimpleRouter()
router.register("api/crud_demo", DataSetModelViewSet)

urlpatterns = [
    path('api/crud_demo/Download/', DataSetModelViewSet.as_view({'get': 'Download'})),
]
urlpatterns += router.urls
