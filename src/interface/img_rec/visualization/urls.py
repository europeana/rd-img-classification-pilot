from django.urls import path

from . import views

urlpatterns = [
    path('', views.datasets, name='index'),
    path('<int:ex_id>/', views.detail_dataset, name='detail_dataset'),
]

#from django.urls import path,re_path
#from datasets.views import DatasetsMainView
#from django.contrib.auth.decorators import login_required
#import visualization.views as dataset_views


# urlpatterns = [
#     path('', DatasetsMainView.as_view()),
#     #path('search', SearchMainView.as_view()),
#     path('<int:ex_id>/', dataset_views.detail_dataset, name='detail_dataset'),
#     path('<int:ex_id>/search', dataset_views.search_dataset, name='search_dataset'),
#     path('<int:dataset>/<category>/', dataset_views.category_view, name='detail_category'),
#     path('<int:dt_id>/<str:category>/<int:img_id>/', dataset_views.detail_image, name='detail_image'),
# ]