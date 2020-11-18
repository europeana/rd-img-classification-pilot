import sys
sys.path.append('/home/jcejudo/rd-europeana-python-api')

from europeana.api import EuropeanaAPI

from europeana.edm import *
from europeana.utils import url2img

def main():

  eu = EuropeanaAPI('api2demo')

  r = eu.search('*', n = 10,skos_concept= 'http%3A%2F%2Fdata.europeana.eu%2Fconcept%2Fbase%2F6', what='painting',sort = {'term':'score','order':'asc'})
  
  print(r.success)

  if r.success:

    for i,edm in enumerate(r.edm_items):

      if edm.title:
        print(edm.title.lang)

main()
