"""
Django settings for snrt_dashboard_project.
"""

from pathlib import Path
from decouple import config, Csv
import os

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# ---- Security / Env ----
SECRET_KEY = config("SECRET_KEY", default="insecure-default-key")
DEBUG = config("DEBUG", default=False, cast=bool)
ALLOWED_HOSTS = config("ALLOWED_HOSTS", cast=Csv(), default=["127.0.0.1", "localhost"])

# ---- Localization ----
LANGUAGE_CODE = "en-us"
TIME_ZONE = config("TIME_ZONE", default="UTC")
USE_I18N = True
USE_TZ = True

# ---- Installed apps ----
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "dashboard",
]

# ---- Middleware ----
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "snrt_dashboard_project.urls"

# ---- Templates ----
# Make sure your custom login template is discoverable at:
# BASE_DIR / templates / registration / login.html
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],   # <— important so /templates is used
        "APP_DIRS": True,                   # also loads dashboard/templates/*
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "snrt_dashboard_project.wsgi.application"

# ---- Database (Django ORM) ----
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ---- External (Mongo) ----
MONGO_URI = config("MONGO_URI", default="mongodb://localhost:27017/")

# ---- Static & Media ----
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
MEDIA_URL = "/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, "media")

# ---- Auth redirects ----
# Names correspond to the routes provided by django.contrib.auth.urls
LOGIN_URL = "login"                   # -> /accounts/login/
LOGIN_REDIRECT_URL = "dashboard"      # after successful login
LOGOUT_REDIRECT_URL = "login"         # redirect to login after logout

# ---- ML / Predictions ----
# Path to the saved scikit-learn pipeline (created in your notebook)
# You can override via env var ML_MODEL_PATH if you put it elsewhere.
ML_MODEL_PATH = Path(
    config("ML_MODEL_PATH", default=str(BASE_DIR / "logit_site_has_station.pkl"))
)

# ---- Defaults ----
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
