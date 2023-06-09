# Create your views here.
from django.http import FileResponse, Http404

from dataset.models import CrudDemoModel
from dataset.serializers import CrudDemoModelSerializer, CrudDemoModelCreateUpdateSerializer
from dataset import Rf_Classifier_Plus
from dvadmin.utils.json_response import SuccessResponse
from dvadmin.utils.viewset import CustomModelViewSet

from django.http import HttpResponse
from wsgiref.util import FileWrapper
import tempfile
import zipfile

from manage import model_number


class CrudDemoModelViewSet(CustomModelViewSet):
    """
    list:查询
    create:新增
    update:修改
    retrieve:单例
    destroy:删除
    """
    queryset = CrudDemoModel.objects.all()
    serializer_class = CrudDemoModelSerializer
    create_serializer_class = CrudDemoModelCreateUpdateSerializer
    update_serializer_class = CrudDemoModelCreateUpdateSerializer
    # filter_fields = ['goods', 'goods_price']
    # search_fields = ['goods']

    # 分析结果 下载
    def Download(self, request):
        try:
            response = FileResponse(open('./media/data/classifyResults.json', 'rb'))
            response['content_type'] = "application/octet-stream"
            response['Content-Disposition'] = 'attachment; filename=classifyResults.json'
            return response
        except Exception:
            raise Http404

    # 模型训练
    def trainDataSet(self, request):
        # file = self.initial_data.get('file')
        file = request.data.get('file')
        Rf_Classifier_Plus.classify(file)
        return SuccessResponse(data=[], msg="模型训练成功")


    # 模型文件下载
    def model_download(self, request):
        file_path_list = []
        # 根据全局变量model_number设置的生成随机森林pdf数量循环添加到file_path
        for i in range(model_number):
            file_path_list.append({'file_name': 'tree_estimators_' + str(i) + '.pdf', 'file_path': './media/treepdf/tree_estimators_' + str(i) + '.pdf'},)

        temp = tempfile.TemporaryFile()
        archive = zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED)
        for file_path_dict in file_path_list:
            file_path = file_path_dict.get('file_path', None)
            file_name = file_path_dict.get('file_name', None)
            archive.write(file_path, file_name)     # TODO need check file exist or not

        archive.close()
        lenth = temp.tell()
        temp.seek(0)

        wrapper = FileWrapper(temp)

        # Using HttpResponse
        response = HttpResponse(wrapper, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=archive.zip'
        response['Content-Length'] = lenth  # temp.tell()
        return response

