from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse
from django.template import loader

from .models import DatasetModel, ImageModel

from django.shortcuts import render, get_object_or_404, redirect, reverse


from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


import sys
sys.path.append('../../self_supervised')

from call_api import ModelAPI



def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def image_view(request):
    api = ModelAPI('img_recommendation',port = 5000)
    context = {}
    return render(request, 'visualization/image.html',context)

def detail_dataset(request,ex_id):

    dataset = get_object_or_404(DatasetModel, pk=ex_id)
    img_list = ImageModel.objects.filter(dataset = ex_id)

    # get dataset

    # get images

    page = request.GET.get('page', 1)

    paginator = Paginator(img_list, 30)
    try:
        img_list = paginator.page(page)
    except PageNotAnInteger:
        img_list = paginator.page(1)
    except EmptyPage:
        img_list = paginator.page(paginator.num_pages)

    # Get the index of the current page
    index = img_list.number - 1  # edited to something easier without index
    # This value is maximum index of your pages, so the last page - 1
    max_index = len(paginator.page_range)
    # You want a range of 7, so lets calculate where to slice the list
    start_index = index - 3 if index >= 3 else 0
    end_index = index + 3 if index <= max_index - 3 else max_index
    # Get our new page range. In the latest versions of Django page_range returns 
    # an iterator. Thus pass it to list, to make our slice possible again.
    page_range = list(paginator.page_range)[start_index:end_index]
    context = {}
    context.update({'page_range':page_range,'img_list':img_list,'dataset':dataset})

    print(len(img_list))


    #context = {'dataset':dataset,'img_list':img_list}
    return render(request, 'visualization/dataset.html',context) 


def datasets(request):

    dataset_list = DatasetModel.objects.all()
    context = {'dataset_list':dataset_list}

    return render(request, 'visualization/datasets.html',context)

    #context.update({'nbar':'datasets','logged':True})

    # dataset_list = DatasetModel.objects.all()
    # for dataset in dataset_list:

    #     img_list = ImageModel.objects.filter(dataset = dataset)


    # #api = ModelAPI('img_recommendation',port = 5000)

    # #data = api.get_data()

    # #url_list = data['urls']
    # #uri_list = data['uris']

    # img_list = [{'url':url,'uri':uri} for url,uri in zip(url_list,uri_list)]
    
    #print(image_list[0])

    # page = request.GET.get('page', 1)

    # paginator = Paginator(img_list, 30)
    # try:
    #     img_list = paginator.page(page)
    # except PageNotAnInteger:
    #     img_list = paginator.page(1)
    # except EmptyPage:
    #     img_list = paginator.page(paginator.num_pages)

    # # Get the index of the current page
    # index = img_list.number - 1  # edited to something easier without index
    # # This value is maximum index of your pages, so the last page - 1
    # max_index = len(paginator.page_range)
    # # You want a range of 7, so lets calculate where to slice the list
    # start_index = index - 3 if index >= 3 else 0
    # end_index = index + 3 if index <= max_index - 3 else max_index
    # # Get our new page range. In the latest versions of Django page_range returns 
    # # an iterator. Thus pass it to list, to make our slice possible again.
    # page_range = list(paginator.page_range)[start_index:end_index]
    # context = {}
    # context.update({'page_range':page_range,'img_list':img_list})



    
    # for img in  api.image_list():
    #     print(img)












    # if request.method == 'POST':
    #     if 'back' in request.POST:
    #         return redirect(f'/datasets/{dataset}')

    # dataset = get_object_or_404(DatasetModel, pk=dataset)

    # img_obj_list = ImageModel.objects.filter(dataset = dataset.id,category = category)[:]

    # img_list = []
    # for img_obj in img_obj_list:
    #     id_enc = img_obj.filename.replace('.jpg','')
    #     europeana_id = id_enc.replace('[ph]','/')
    #     img_list.append({'url':img_obj.img_url,'id':europeana_id,'id_enc':id_enc,'img_id':img_obj.id})

    # page = request.GET.get('page', 1)

    # paginator = Paginator(img_list, 30)
    # try:
    #     img_list = paginator.page(page)
    # except PageNotAnInteger:
    #     img_list = paginator.page(1)
    # except EmptyPage:
    #     img_list = paginator.page(paginator.num_pages)

    # # Get the index of the current page
    # index = img_list.number - 1  # edited to something easier without index
    # # This value is maximum index of your pages, so the last page - 1
    # max_index = len(paginator.page_range)
    # # You want a range of 7, so lets calculate where to slice the list
    # start_index = index - 3 if index >= 3 else 0
    # end_index = index + 3 if index <= max_index - 3 else max_index
    # # Get our new page range. In the latest versions of Django page_range returns 
    # # an iterator. Thus pass it to list, to make our slice possible again.
    # page_range = list(paginator.page_range)[start_index:end_index]
    #context = {}

    #context.update({'page_range':page_range,'img_list':img_list,'category':category,'dataset_id': dataset.id})
    