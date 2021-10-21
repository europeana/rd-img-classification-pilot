from django.contrib import admin

# Register your models here.
#admin.site.register(DatasetModel)
#from img_rec.settings import MEDIA_ROOT
from django.core.files.storage import FileSystemStorage
import os
from .models import ImageModel, DatasetModel
from shutil import copyfile

from .forms import DatasetForm

class MyAdminView(admin.ModelAdmin):
       def save_model(self, request, obj, form, change):

        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():

            obj = DatasetModel() 
            obj.name = form.cleaned_data['name']
            obj.images_path = form.cleaned_data['images_path']

            dataset_folder_name = os.path.split(obj.images_path)[1]
            cat_list = os.listdir(obj.images_path)

            obj.categories = ' '.join(cat_list)
            obj.save()

            for cat in cat_list:
                cat_path = os.path.join(obj.images_path,cat)

                for fname in os.listdir(cat_path):
                    uploaded_file_url = f'/static/{dataset_folder_name}/{cat}/{fname}'
                    img_obj = ImageModel(filename = fname, img_url = uploaded_file_url) 
                    img_obj.save()
                    img_obj.dataset.add(obj)
            
        else:
            form = DatasetForm()


        super(MyAdminView, self).save_model(request, obj, form, change)


admin.site.register(DatasetModel, MyAdminView)
admin.site.register(ImageModel)

