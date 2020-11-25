from rest_framework import serializers
from WebAnalyzer.models import *


class VideoSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = VideoModel
        fields = ('video', 'video_id', 'token', 'uploaded_date', 'updated_date')
        read_only_fields = ('token', 'uploaded_date', 'updated_date')


class FrameSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = FrameModel
        fields = ('frame', 'info', 'token','uploaded_date', 'updated_date', 'result')
        read_only_fields = ('frame', 'token', 'uploaded_date', 'updated_date', 'result')