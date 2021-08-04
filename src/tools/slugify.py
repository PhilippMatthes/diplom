import re


def slugify(s):
    s = str(s).strip().replace(' ', '_').lower()
    return re.sub(r'(?u)[^-\w.]', '', s)
