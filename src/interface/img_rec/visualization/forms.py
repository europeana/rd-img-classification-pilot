from django import forms
from django.db import models

from .models import DatasetModel
class DatasetForm(forms.ModelForm):
        def __init__(self, *args, **kwargs):
            # first call parent's constructor
            super(DatasetForm, self).__init__(*args, **kwargs)
            #self.fields['annotations_path'].required = False
            #self.fields['annotations_upload'].required = False
            self.fields['categories'].required = False

            if 'img_list' in self.fields.keys():
                self.fields['img_list'].required = False

            self.fields['name'].widget.attrs.update({'class' : 'short-width'})
            self.fields['images_path'].widget.attrs.update({'class' : 'long-width'})
            #self.fields['problem_type'].widget.attrs.update({'class' : 'short-width'})
            #self.fields['annotations_path'].widget.attrs.update({'class' : 'long-width'})

        class Meta:
            model = DatasetModel
            fields = [
                'name',
                'images_path',
                'categories'
                #'problem_type',
                #'annotations_path',
                #'annotations_upload',
                #'is_public',
                ]
  

