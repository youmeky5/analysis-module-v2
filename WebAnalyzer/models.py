import os

from django.db import models

# Create your models here.
from rest_framework import exceptions

from AnalysisModule import settings
from AnalysisModule.config import DEBUG
from WebAnalyzer.tasks import analyzer_by_path
from WebAnalyzer.utils import filename
from django_mysql.models import JSONField
import cv2
import json
import ast


class VideoModel(models.Model):
    video = models.FileField(upload_to=filename.uploaded_date)
    video_id = models.IntegerField(null=False)
    token = models.AutoField(primary_key=True)
    uploaded_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        super(VideoModel, self).save(*args, **kwargs)
        super(VideoModel, self).save()


class FrameModel(models.Model):
    frame = models.ImageField(max_length=255)
    info = models.TextField(null=False)
    token = models.AutoField(primary_key=True)
    uploaded_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    result = JSONField(null=True)

    def save(self, *args, **kwargs):
        super(FrameModel, self).save(*args, **kwargs)
        print(self.info)
        info = ast.literal_eval(self.info)
        frame_info = info["frame_info"]
        video_info = info["video_info"]
        video = VideoModel.objects.filter(token=video_info["token"])[0]
        video_capture = cv2.VideoCapture(video.video.path)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_info['frame_number'] - 1)
        ret, frame = video_capture.read()
        if ret:
            frame_dir = "/workspace/" + settings.MEDIA_URL + str(video.video.name).split(".")[0]
            frame_path = frame_dir + "/{}.jpg".format(int(frame_info['frame_number']))
            if not os.path.exists(frame_path) :
                os.mkdir(frame_dir)
            cv2.imwrite(frame_path, frame)
            self.frame = frame_path
            self.frame.name = str(video.video.name).split(".")[0] + "/{}.jpg".format(int(frame_info['frame_number']))
            if DEBUG:
                task_get = ast.literal_eval(str(analyzer_by_path(frame_path)))
            else:
                task_get = ast.literal_eval(str(analyzer_by_path.delay(frame_path).get()))

            self.result = task_get
        super(FrameModel, self).save()
