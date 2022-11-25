from django.contrib import admin

# Register your models here.

from .models import features,preprocessing,ml_model,registration


admin.site.register(features)
admin.site.register(preprocessing)
admin.site.register(ml_model)
admin.site.register(registration)
