from django import template

register = template.Library()

@register.filter
def get_item(array, key):
    return array[key]