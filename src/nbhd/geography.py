'''Geographical analysis.'''

from time import time

import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
import networkx as nx

from shapely.geometry import (Point, LineString, Polygon, MultiPolygon)
from shapely.ops import polygonize
from mapclassify import greedy

from .data import Base
from .geometry import tessellate, cellularize, trim, pointbox

def get_communities(db,
                    polygon,
                    footprint_threshold=250,
                    res_length_threshold=50,
                    short_threshold=50,
                    node_distance=20,
                    min_community_size = 1):
    

    print(f'Starting to find neighbourhoods in polygon centred at {polygon.centroid}')
    # get nearest properties and buildings for given geometry
    df = db.knn('properties', 'buildings', polygon)

    print(f'Collected {len(df)} properties...')
    # drop properties that are not in buildings
    df = df.loc[df.dist==0]
    print(f'... of which {len(df)} are in buildings')
    

    # eliminate non-residential buildings
    building_counts = dict(df.buildings_id.value_counts())
    df['building_counts'] = df.buildings_id.apply(lambda x: building_counts.get(x,0))
    df['footprint_area'] = gpd.GeoSeries.from_wkb(df.buildings_geometry).area
    df['footprint_area_per_uprn'] = df.footprint_area / df.building_counts
    df['residential_building'] = df[
        'footprint_area_per_uprn'] < footprint_threshold
    df1 = df.loc[df.residential_building].copy()

    print('Getting streets...')
    # get streets 
#     bdgs_multiplygn = MultiPolygon(
#         list(gpd.GeoSeries.from_wkb(df1.buildings_geometry).geometry.values))
    df2 = db.knn('properties', 'roads', 
                 polygon, 
                 t2_columns=['"startNode"', '"endNode"'])
    df2.loc[df2.properties_id.apply(lambda x: x in df1.properties_id)]

    df2 = gpd.GeoDataFrame(df2, geometry=gpd.GeoSeries.from_wkb(df2.roads_geometry))
    df2['length'] = df2.geometry.length

    # 3 establish whether roads are residential
    street_counts = dict(df2.roads_id.value_counts())
    df2['street_counts'] = df2.roads_id.apply(
        lambda x: street_counts.get(x, 0))
    df2['street_length_per_uprn'] = df2.length / df2.street_counts
    df2['residential_street'] = df2.street_length_per_uprn < res_length_threshold
    residential = dict(zip(df2.roads_id, df2.residential_street))
    df2['residential'] = df2.roads_id.apply(
        lambda x: residential.get(x, False))
    df2['short_street'] = df2.length < short_threshold
    df2['res_or_short'] = df2.residential | df2.short_street
    df3 = df2.loc[df2.res_or_short].copy()

    print('Snapping nodes...')
    # 4 treat nearby nodes as equivalent
    translator = nn_translator(db, 'nodes',
        polygon, node_distance)

    edges = df3.loc[~df3.duplicated()].copy()
    edges['translated_start'] = edges.startNode.apply(
        lambda x: translator.get(x, x))
    edges['translated_end'] = edges.endNode.apply(
        lambda x: translator.get(x, x))

    print('Finding connected graphs...')
    # 5 find connected networks of residential streets
    g = nx.from_pandas_edgelist(edges, 'translated_start',
                                'translated_end', True)
    subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
    sgs = [sg for sg in subgraphs if len(sg) > min_community_size]

    print('Labelling communities.')
    # 6 add community labels
    communities = dict()
    for i in range(len(sgs)):
        communities[str(i+1).zfill(2)] = list(
            nx.get_edge_attributes(sgs[i], 'roads_id').values())
    communities_key = {
        value: key
        for key, value_list in communities.items() for value in value_list
    }
    df3['community'] = df3.roads_id.apply(
        lambda x: communities_key.get(x, None))
    
    # merge property~streets df (df3) with property~buildings df (df1)
    merged_df = df3.merge(df1, on=['properties_id','properties_geometry'], how='inner')

    return merged_df, sgs

def square_plot(x, y, radius, db, cmap='Paired', alpha=0.2):
    
    # define point
    p = Point(x, y)
    
    # get bounding box
    b = pointbox(p, radius)
    boundary = gpd.GeoDataFrame(geometry=gpd.GeoSeries(b.boundary))
    
    # get data within bounding box
    roads = db.intersects('roads', b)
    buildings = db.intersects('buildings', b)
    properties = db.within('properties', MultiPolygon(buildings.geometry.values))
    
    roads = roads.loc[roads.road_function != 'Secondary Access Road']
    
    # tessellate
    tiles = tessellate([roads, boundary])
    tiles['c'] = greedy(tiles)
    
    # trim
    roads.geometry = roads.geometry.apply(lambda x: trim(x, b))
    buildings.geometry = buildings.geometry.apply(lambda x: trim(x, b))
    tiles.geometry = tiles.geometry.apply(lambda x: trim(x,b))
    properties = properties.loc[properties.geometry.within(b)]
    
    # plot
    f,ax = plt.subplots(figsize=(12,12))
    tiles.plot('c', cmap=cmap, alpha=alpha, ax=ax)
    roads.plot(ax=ax, color='tab:brown', linewidth=6)
    buildings.boundary.plot(ax=ax, color='k')
    buildings.plot(ax=ax, color='grey')
    properties.plot(color='k', ax=ax)
    boundary.plot(ax=ax, color='k', linewidth=5)
    ax.set_axis_off()
    return ax, [roads, buildings, properties, boundary, tiles]

def nn_translator(db, table, polygon, max_distance):
    nearest_nodes = db.knn('nodes','nodes',polygon,
                           k=10)
    nearest_nodes = nearest_nodes.loc[nearest_nodes.dist<max_distance]

    col_names = list(nearest_nodes.columns)
    col_names[0], col_names[2] = 'first', 'second'
    nearest_nodes.columns = col_names

    g = nx.from_pandas_edgelist(nearest_nodes,'first','second')
    subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
    translator = {n: list(sorted(g.nodes))[0] for g in subgraphs for n in g.nodes}
    return translator

class Neighbourhood:
    '''
    Theoretically-informed and data-driven neighbourhood analysis.
    '''
    def __init__(self, easting=338172, northing=391744, db=None, load=True):
        '''Begin with a point on the British National Grid.'''

        self.db = db
        # find enclosure from database

        enclosures = db.within('bigtilesa', 0, easting, northing)
        if len(enclosures) == 1:
            self.geom = enclosures.geometry[0]
        else:
            print('boundary point!')
            self.geom = MultiPolygon(list(enclosures.geometry))

        self.boundary = gpd.GeoSeries(self.geom.boundary)

        if load:
            self.get_data()

    def get_communities(self,
                        footprint_threshold=250,
                        res_length_threshold=20,
                        short_threshold=20,
                        min_community_size=0,
                        node_distance=5):

        # nearest roads ~ 'slimroads' is without motorways and secondary roads
        nr_roads = self.db.nearest_neighbours('slimroads',
                                              self.geom.buffer(10))
        # nearest buildings
        nr_buildings = self.db.nearest_neighbours('openmaplocal',
                                                  self.geom.buffer(1))
        # merge on UPRN
        df = nr_buildings.merge(nr_roads,
                                on=['UPRN', 'uprn_geometry'],
                                how='inner',
                                suffixes=('_building', '_street'))

        # 1 eliminate non-building properties : distance to building must == 0
        df1 = df.loc[df.dist_building == 0].copy()

        # 2 eliminate non-residential buildings
        building_counts = dict(df1.id_building.value_counts())
        df1['building_counts'] = df1.id_building.apply(
            lambda x: building_counts.get(x, 0))
        df1['footprint_area'] = gpd.GeoSeries(df1.geometry_building).area
        df1['footprint_area_per_uprn'] = df1.footprint_area / df1.building_counts
        df1['residential_building'] = df1[
            'footprint_area_per_uprn'] < footprint_threshold
        df2 = df1.loc[df1.residential_building].copy()

        # 3 establish whether roads are residential
        street_counts = dict(df2.id_street.value_counts())
        df2['street_counts'] = df2.id_street.apply(
            lambda x: street_counts.get(x, 0))
        df2['street_length_per_uprn'] = df2.length / df2.street_counts
        df2['residential_street'] = df2.street_length_per_uprn < res_length_threshold
        residential = dict(zip(df2.id_street, df2.residential_street))
        df['residential'] = df.id_street.apply(
            lambda x: residential.get(x, False))
        df['short_street'] = df.length < short_threshold
        df['res_or_short'] = df.residential | df.short_street
        df3 = df.loc[df.res_or_short].copy()

        # 4 treat nearby nodes as equivalent
        translator = self.db.get_nearest_nodes_translator(
            self.geom, node_distance)
        edges = df3.loc[~df3.duplicated()].copy()
        edges['translated_start'] = edges.startNode.apply(
            lambda x: translator.get(x, x))
        edges['translated_end'] = edges.endNode.apply(
            lambda x: translator.get(x, x))

        # 5 find connected networks of residential streets
        g = nx.from_pandas_edgelist(edges, 'translated_start',
                                    'translated_end', True)
        subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
        sgs = [sg for sg in subgraphs if len(sg) > min_community_size]

        # 6 add community labels
        communities = dict()
        for i in range(len(sgs)):
            communities[str(i).zfill(2)] = list(
                nx.get_edge_attributes(sgs[i], 'id_street').values())
        communities_key = {
            value: key
            for key, value_list in communities.items() for value in value_list
        }

        df['community'] = df.id_street.apply(
            lambda x: communities_key.get(x, None))
        self.df = df
        return

    def get_data(self,
                 wanted=('roads', 'uprn', 'buildings'),
                 plus=None,
                 silent=True):

        t0 = time()

        wanted_list = list(wanted)
        if plus:
            wanted_list.extend(plus)

        table = {
            'roads': 'openroads',
            'uprn': 'openuprn',
            'buildings': 'openmaplocal',
            'rail': 'railways',
            'rivers': 'rivers'
        }

        for w in wanted_list:
            # need to buffer this slightly
            setattr(self, w, self.db.contains(table[w],
                                              self.geom.buffer(1).wkt))

        t1 = time()
        t = t1 - t0
        print(('Getting data took'),
              (f'{int(t//60)} minutes, {int(t%60)} seconds.'))

    def plot(self, figsize_x=12, figsize_y=12, cmap='Set2'):

        _fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
        self.buildings.geometry.boundary.plot(color='k', linewidth=0.6, ax=ax)
        self.uprn.plot('street', ax=ax, markersize=5, cmap=cmap)
        self.roads.plot('id', ax=ax, linewidth=2, cmap=cmap)
        self.boundary.plot(ax=ax, linewidth=1.5, color='k')
        self.ax = ax

    def find_neighbours(self):

        self._neighbour = dict()
        # this probably just needs the right merge
        self._neighbour['streets'] = self.db.nearest_neighbours(
            'roadsnotsec', self.geom.buffer(1))
        _streets = dict(
            zip(self._neighbour['streets'].UPRN,
                self._neighbour['streets'].id))
        self.uprn['street'] = self.uprn.UPRN.apply(lambda x: _streets.get(x))

    def tessellate(self):

        self.tiles = tessellate([
            self.roads,
            gpd.GeoDataFrame(geometry=gpd.GeoSeries(self.geom.boundary))
        ])

    def cellularize(self):
        self.cells = cellularize(self.uprn.geometry, self.geom)


class Community:
    def __init__(self, df, graph_list, label_number):
        
        self.number = label_number
        self.label = str(label_number + 1).zfill(2)
        self.df = df.loc[df.community == self.label]
        self.graph = graph_list[self.number]
        
        self.gdf = dict()
        self.gdf['properties'] = gpd.GeoDataFrame(self.df, 
                                                  geometry=gpd.GeoSeries.from_wkb(
                                                      c.df.properties_geometry))
        self.gdf['buildings'] = gpd.GeoDataFrame(self.df, 
                                                  geometry=gpd.GeoSeries.from_wkb(
                                                      c.df.buildings_geometry))
        self.gdf['roads'] = gpd.GeoDataFrame(self.df, 
                                                  geometry=gpd.GeoSeries.from_wkb(
                                                      c.df.roads_geometry))
        self.stats()
    
    def stats(self):
        self.count = dict()
        self.count['properties'] = len(self.df.properties_id.unique())
        self.count['roads'] = len(self.df.roads_id.unique())
        self.count['buildings'] = len(self.df.buildings_id.unique())
    
    def __repr__(self):
        return str(f'Community {label_number}')
    

class FaceBlock():
    def __init__(self, df, roads_id, db, get_stats=False):
        
        self.db = db
        
        print(f'FaceBlock: {roads_id}')
        self.id = roads_id
        self.total_df = df
        self.df = self.total_df.loc[self.total_df.roads_id == roads_id]
        self.df = gpd.GeoDataFrame(self.df, geometry=gpd.GeoSeries.from_wkb(self.df.properties_geometry))
        self.properties = self.df.properties_id.unique()
        self.buildings_df = self.df[~self.df.buildings_id.duplicated()]
        self.buildings_df = gpd.GeoDataFrame(self.buildings_df,
                                            geometry=gpd.GeoSeries.from_wkb(self.buildings_df.buildings_geometry))
        self.buildings = self.df.buildings_id.unique()
        self.roads_df = self.df[~self.df.roads_id.duplicated()]
        self.roads_df = gpd.GeoDataFrame(self.roads_df,
                                            geometry=gpd.GeoSeries.from_wkb(self.roads_df.roads_geometry))       
        assert len(self.roads_df.roads_id.unique() == 1)
       
        print('Finding neighbours...')
        
        self.startNode = self.roads_df.startNode.values[0]
        self.endNode = self.roads_df.endNode.values[0]
        self.neighbours = dict()
        self.neighbours['start'] = list(self.total_df.loc[
            (self.total_df.startNode==self.startNode) & (self.total_df.roads_id != self.id)].roads_id.unique()) + \
        list(self.total_df.loc[
            (self.total_df.endNode==self.startNode) & (self.total_df.roads_id != self.id)].roads_id.unique())
        self.neighbours['end'] = list(self.total_df.loc[
            (self.total_df.startNode==self.endNode) & (self.total_df.roads_id != self.id)].roads_id.unique()) + \
        list(self.total_df.loc[
            (self.total_df.endNode==self.endNode) & (self.total_df.roads_id != self.id)].roads_id.unique())

        print(self.neighbours)
        
        if get_stats:
            print('Getting summary statistics...')
            self.stats()
       
    def __repr__(self):
#         if not hasattr(self, 'stats_df'):
#             self.stats()
#         display(self.stats_df)
        if not hasattr(self, 'road_name'):
            self.get_name()
        return f"{self.road_name} ({self.road_function}): {self.count['properties']} properties."
         
    def get_name(self):
        self.road_name, self.road_function = self.db.query(
        f"SELECT name1, road_function FROM roads WHERE id = '{self.id}'").values[0]
        
    def stats(self):
        self.count = dict()
        self.count['segment_length'] = self.roads_df['length'].values[0]
        self.count['segment_length_per_property'] = self.roads_df['street_length_per_uprn'].values[0]
        self.count['properties'] = len(self.properties)
        self.count['buildings'] = len(self.buildings)
        self.count['neighbouring_face_blocks'] = len(self.neighbours['start']) + len(self.neighbours['end'])
        block_prop_count = dict(self.df.value_counts('buildings_id'))
        self.buildings_df['block_prop_count'] = self.buildings_df.buildings_id.apply(lambda x: block_prop_count[x])
        self.buildings_df['block_pct'] = self.buildings_df.block_prop_count / self.buildings_df.building_counts
        self.buildings_df['block_footprint'] = self.buildings_df.footprint_area * self.buildings_df.block_pct
        self.count['block_building_footprint'] = self.buildings_df.block_footprint.sum()
        self.count['median_property_footprint'] = self.buildings_df.footprint_area_per_uprn.median()
        self.count['mean_property_footprint'] = self.buildings_df.footprint_area_per_uprn.mean()
        self.stats_df = pd.DataFrame(self.count, index=['statistics'])
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12,12))
        self.buildings_df.boundary.plot(color='k', ax=ax)
        self.buildings_df.plot('block_pct', cmap='binary', ax=ax, alpha=0.5)
        self.roads_df.plot(color='k', ax =ax, linewidth=10)
        self.df.plot(color='k',ax=ax)
        ax.set_axis_off()
        self.fig = ax.figure