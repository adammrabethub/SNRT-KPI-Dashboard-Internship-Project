from django.contrib import admin
from django.urls import path, include
from django.contrib.auth.views import LogoutView
from dashboard import views as dashboard_views

urlpatterns = [
    path("admin/", admin.site.urls),

    # 1) Put the custom logout FIRST so it wins the match:
    path("accounts/logout/", LogoutView.as_view(next_page="login"), name="logout"),

    # 2) Then include all the built-in auth routes (login, password reset, etc.)
    path("accounts/", include("django.contrib.auth.urls")),

    # Dashboard
    path("", dashboard_views.dashboard_view, name="dashboard"),
    path("kpi-data/", dashboard_views.kpi_data, name="kpi_data"),
    path("", include("dashboard.urls")),
    path("download-excel/", dashboard_views.download_excel, name="download_excel"),

]
