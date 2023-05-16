from crud_demo.models import DataSetModel
from dvadmin.utils.serializers import CustomModelSerializer


class DataSetModelSerializer(CustomModelSerializer):
    """
    序列化器
    """

    class Meta:
        model = DataSetModel
        fields = "__all__"


class DataSetModelCreateUpdateSerializer(CustomModelSerializer):
    """
    创建/更新时的列化器
    """

    class Meta:
        model = DataSetModel
        fields = '__all__'