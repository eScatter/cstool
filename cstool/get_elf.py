"""
Get Energy Loss Function data directly from web database, serving X-ray
refraction indices [1]_ at http://henke.lbl.gov/optical_constants.

.. [1] B.L. Henke, E.M. Gullikson, and J.C. Davis. X-ray interactions:
    photoabsorption, scattering, transmission, and reflection at E=50-30000 eV,
    Z=1-92, Atomic Data and Nuclear Data Tables Vol. 54 (no.2),
    181-342 (July 1993).
"""

import requests
from html.parser import HTMLParser
import re


class GetRefreshURL(HTMLParser):
    def __init__(self, root=''):
        super(GetRefreshURL, self).__init__()
        self.url = None
        self.root = root
        self.rgx = re.compile('URL=(.*)')

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        if tag == 'meta' and attr_dict.get('http-equiv') == 'REFRESH':
            print(attr_dict.get('content'))
            match = re.search('URL\=(.*)', attr_dict.get('content'))
            if not match:
                return
            partial_url = match.group(1)
            self.url = self.root + partial_url


def lbl_ior(formula, E_min, E_max, N_pts, density=-1):
    url = "http://henke.lbl.gov/cgi-bin/getdb.pl"

    form_data = {
        "Formula": formula,
        "Density": str(density),
        "Min": str(E_min),
        "Max": str(E_max),
        "Npts": str(N_pts),
        "Scan": "Energy",
        "Output": "Text File",
        "submit": "Submit Request"
    }

    response = requests.post(url, data=form_data)

    p = GetRefreshURL(root="http://henke.lbl.gov")
    p.feed(response.content.decode())

    if not p.url:
        return None

    response = requests.get(p.url)
    return response.content.decode()
