import hashlib
import json

from rest_framework import serializers

from application import dispatch
from dvadmin.system.models import FileList
from dvadmin.utils.serializers import CustomModelSerializer
from dvadmin.utils.viewset import CustomModelViewSet
from dataset import Model_Invocation


class FileSerializer(CustomModelSerializer):
    url = serializers.SerializerMethodField(read_only=True)

    def get_url(self, instance):
        # return 'media/' + str(instance.url)
        return instance.file_url or (f'media/{str(instance.url)}')

    class Meta:
        model = FileList
        fields = "__all__"

    def create(self, validated_data):
        file_engine = dispatch.get_system_config_values("fileStorageConfig.file_engine") or 'local'
        file_backup = dispatch.get_system_config_values("fileStorageConfig.file_backup")
        file = self.initial_data.get('file')

        content_type = file.content_type
        # 算法分析处理
        if content_type == "text/csv":
            print("running...")
            # Rf_Classifier_Plus.classify()
            # 直接调用训练好的模型
            macro_P, macro_R, macro_F1, LABEL0, LABEL1, LABEL2, LABEL3, LABEL4, LABEL5, precision_i, recall_i, dictionary = Model_Invocation.test_set(
                file)
            print("done")
            dict = {'macro_P': macro_P, 'macro_R': macro_R, 'macro_F1': macro_F1,
                    'LABEL0': LABEL0, 'LABEL1': LABEL1, 'LABEL2': LABEL2, 'LABEL3': LABEL3, 'LABEL4': LABEL4,
                    'LABEL5': LABEL5,
                    'precision_i': precision_i, 'recall_i': recall_i}
            # # 转json对象，以便前台处理
            validated_data['dict'] = json.dumps(dict)

        # 存储json结果文件
        # with open('./media/data/classifyResults.json', 'w') as json_file:
        #     json.dump(dictionary, json_file, indent=4)

            # dictionary = {}
            # keys = []
            # values = []
            # keys.append("1")
            # values.append(5)
            #
            # for key, value in zip(keys, values):
            #     dictionary[key] = value
            # with open('./media/data/classifyResults.json', 'r') as json_file:
            #     validated_data['json_result'] = json_file.read()

        file_size = file.size
        validated_data['name'] = file.name
        validated_data['size'] = file_size
        md5 = hashlib.md5()
        for chunk in file.chunks():
            md5.update(chunk)
        validated_data['md5sum'] = md5.hexdigest()
        validated_data['engine'] = file_engine
        validated_data['mime_type'] = file.content_type
        if file_backup:
            validated_data['url'] = file
        if file_engine == 'oss':
            from dvadmin_cloud_storage.views.aliyun import ali_oss_upload
            file_path = ali_oss_upload(file)
            if file_path:
                validated_data['file_url'] = file_path
            else:
                raise ValueError("上传失败")
        elif file_engine == 'cos':
            from dvadmin_cloud_storage.views.tencent import tencent_cos_upload
            file_path = tencent_cos_upload(file)
            if file_path:
                validated_data['file_url'] = file_path
            else:
                raise ValueError("上传失败")
        else:
            validated_data['url'] = file
        # 审计字段
        try:
            request_user = self.request.user
            validated_data['dept_belong_id'] = request_user.dept.id
            validated_data['creator'] = request_user.id
            validated_data['modifier'] = request_user.id
        except:
            pass
        return super().create(validated_data)


class FileViewSet(CustomModelViewSet):
    """
    文件管理接口
    list:查询
    create:新增
    update:修改
    retrieve:单例
    destroy:删除
    """
    queryset = FileList.objects.all()
    serializer_class = FileSerializer
    filter_fields = ['name', ]
    permission_classes = []
