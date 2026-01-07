# dashboard/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Main dashboard
    path("", views.dashboard_view, name="dashboard"),

    # JSON APIs for charts & detail views
    path("api/kpi-data/", views.kpi_data, name="kpi_data"),
    path("api/station-equipment-detail/", views.station_equipment_detail, name="station_equipment_detail"),

    # Prediction APIs
    path("api/site-predictions/", views.site_predictions, name="site_predictions"),  # JSON + CSV
    path("api/site-predict-simulate/", views.site_predict_simulate, name="site_predict_simulate"),  # NEW — what-if
    path("api/site-predict-meta/", views.site_predict_meta, name="site_predict_meta"),  # NEW — model meta

    # Data health check
    path("data-health/", views.data_health, name="data_health"),

    # Downloads
    path("download/excel/", views.download_excel, name="download_excel"),
]
