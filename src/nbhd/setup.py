'''
Simplify the process of downloading and extracting Ordnance Survey data.
'''

import zipfile

import fiona
import pandas as pd
import geopandas as gpd
import requests


class OSDownloader():
    def __init__(self,
                 download_immediately=True,
                 api_url='https://api.os.uk/downloads/v1/products'):

        response = requests.get(api_url)
        products = pd.DataFrame(response.json())
        products['download'] = products.url.apply(
            lambda x: requests.get(x).json()['downloadsUrl'])
        # products['download2'] = products.download.apply(lambda x: requests.get(x).json())
        products['formats'] = products.download2.apply(
            lambda x: [e['format'] for e in x])
        products['geopackage'] = products.formats.apply(
            lambda x: 'GeoPackage' in x)
        products['geopackage_url'] = products.apply(
            lambda x:
            [e['url'] for e in x.download2 if e['format'] == 'GeoPackage'][0]
            if x.geopackage else None,
            axis=1)

        def download_row(row, _format):
            print(row.id)
            try:
                [url] = [
                    e['url'] for e in row.download2
                    if e['format'].split(' ')[-1] == _format
                ]
                with open(f'../data/{row.id}{_format}.zip',
                          'wb') as downloading:
                    downloading.write(requests.get(url).content)
                    downloading.close()
                return "Done!"
            except Exception as e:
                print(e)

        products['downloaded'] = products.apply(
            lambda x: download_row(x, 'GeoPackage') if x.geopackage\
            else download_row(x, 'Shapefile'),
            axis=1
        )


class ZippedGpkg():
    def __init__(self, filepath, db_engine):

        self.filepath = filepath
        self.db_engine = db_engine

        self.zf = zipfile.ZipFile(filepath)
        self.unpacked = dict()
        self.namelist = self.zf.namelist()

        try:
            [self.gpkg
             ] = [f for f in self.namelist if f.split('.')[-1] == 'gpkg']

            self.layers = fiona.listlayers(self.zf.open(self.gpkg))
            self.type = 'GeoPackage'
        except:
            print('Not a GeoPackage, perhaps a Shapefile?')
            self.layers = {
                s.split('/')[2].split('.')[0]
                for s in self.namelist if len(s.split('/')) > 2
            }
            self.type = 'Shapefile'

    def unpack(self, layername, db_tablename):

        if self.type == 'GeoPackage':
            gdf = gpd.read_file(self.zf.open(self.gpkg),
                                driver='GPKG',
                                layer=layername)
            gdf.to_postgis(db_tablename, self.db_engine)
        elif self.type == 'Shapefile':
            names = [
                n for n in self.namelist
                if layername in n and n.split('.')[-1] == 'shp'
            ]
            for name in names:
                print(name)
                try:
                    gdf = gpd.read_file(self.zf.open(name))
                    gdf.to_postgis(db_tablename, db_engine)
                except Exception as e:
                    print(e)
