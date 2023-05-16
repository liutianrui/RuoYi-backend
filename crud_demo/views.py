# Create your views here.
import json

from requests import Response

from crud_demo.models import DataSetModel
from crud_demo.serializers import DataSetModelSerializer, DataSetModelCreateUpdateSerializer
from dvadmin.utils.viewset import CustomModelViewSet

from django.http import FileResponse
import os
from django.http import HttpResponse, Http404, FileResponse


class DataSetModelViewSet(CustomModelViewSet):
    """
    list:查询
    create:新增
    update:修改
    retrieve:单例
    destroy:删除
    """
    queryset = DataSetModel.objects.all()
    serializer_class = DataSetModelSerializer
    create_serializer_class = DataSetModelCreateUpdateSerializer
    update_serializer_class = DataSetModelCreateUpdateSerializer
    filter_fields = ['sample_id', 'label']
    search_fields = ['sample_id', 'label']

    def Download(self, request):
        try:
            response = FileResponse(open('./media/data/classifyResults.json', 'rb'))
            response['content_type'] = "application/octet-stream"
            response['Content-Disposition'] = 'attachment; filename=classifyResults.json'
            return response
        except Exception:
            raise Http404
