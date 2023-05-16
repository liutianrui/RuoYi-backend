from django.db import models
# from gunicorn.config import User

# Create your models here.
from dvadmin.utils.models import CoreModel
from django.db import models

class DataSetModel(CoreModel):

    sample_id = models.IntegerField(verbose_name="sample_id")
    feature0 = models.TextField(verbose_name="feature0", blank=True)
    feature1 = models.TextField(verbose_name="feature1", blank=True)
    feature2 = models.TextField(verbose_name="feature2", blank=True)
    feature3 = models.TextField(verbose_name="feature3", blank=True)
    feature4 = models.TextField(verbose_name="feature4", blank=True)
    feature5 = models.TextField(verbose_name="feature5", blank=True)
    feature6 = models.TextField(verbose_name="feature6", blank=True)
    feature7 = models.TextField(verbose_name="feature7", blank=True)
    feature8 = models.TextField(verbose_name="feature8", blank=True)
    feature9 = models.TextField(verbose_name="feature9", blank=True)
    feature10 = models.TextField(verbose_name="feature10", blank=True)
    label = models.IntegerField(verbose_name="label", blank=True)

    class Meta:
        db_table = "dataset"
        verbose_name = '数据集表'
        verbose_name_plural = verbose_name
        ordering = ('-create_datetime',)