"""
This module includes the primary `get_communities()` function, and
several helper functions that are necessary for it.
"""

import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt

from shapely.geometry import Polygon

from nbhd import data


def get_communities(
    db: data.Base,
    polygon: Polygon,
    residential_street_threshold: int = 30,
    short_threshold: int = 10,
    snap_nodes_threshold: int = 5,
    quiet: bool = True,
):
    """
    This is the primary function to get networks of connected residential street communities.

    Returns two GeoDataFrames.

    The first, `streets` is like the dataframe returned by querying the
    `roads` table, but it includes several other calculated features of
    interest: `number_of_properties`, `length_per_property`,
    `residential`, `short`, `short_or_residential`, `community`, and
    `community_size`.

    The second, `properties`, includes the `properties_id` and
    `properties_geometry` columns directly relevant to each property, as
    well as the `roads_id`, `startNode`, `endNode` and `roads_geometry`
    of its nearest road, and the euclidean `dist`(ance) between them, as
    well as its `community` label and `faceblock_community_size`.
    """

    # first we need all the roads in the polygon, so that those without
    # a nearby property don't get missed out.

    if not quiet:
        print("First getting all roads in polygon...")
    all_streets = db.intersects("roads", polygon)
    if not quiet:
        print(f"That is {len(all_streets)} road segments in total!")

    if not quiet:
        print("Now marking whether they are residential...")
    residential_streets, properties = get_residential_streets_and_properties(
        db, polygon, residential_street_threshold
    )
    if not quiet:
        print(
            (
                f"{len(residential_streets.loc[residential_streets.residential])}"
                " are residential."
            )
        )

    streets = pd.concat(
        [all_streets.rename({"id": "roads_id"}, axis=1), residential_streets],
        keys="roads_id",
        join="outer",
    )
    streets["short"] = streets.geometry.length < short_threshold
    streets["short_or_residential"] = streets.residential | streets.short

    if snap_nodes_threshold > 0:
        if not quiet:
            print("Now snapping nearby nodes together...")
        streets = snap_together_nearby_nodes(streets, snap_nodes_threshold, db, polygon)

    if not quiet:
        print("Now labelling connected communities...")
    streets, properties = label_connected_communities(streets, properties)
    community_size_dict = dict(streets.community.value_counts())
    streets["community_size"] = streets.community.apply(
        lambda x: community_size_dict.get(x, 0)
    )
    properties["faceblock_community_size"] = properties.community.apply(
        lambda x: community_size_dict.get(x, 0)
    )

    return streets, properties


def get_residential_streets_and_properties(
    db: data.Base, polygon: Polygon, residential_street_threshold: int
):
    """
    Return dataframe of streets marked `short_or_residential`.
    """
    properties_on_streets_df = get_properties_on_streets(db, polygon)
    properties_on_streets_df["residential"] = mark_residential_properties(
        properties_on_streets_df, db, polygon
    )

    residential_properties_df = properties_on_streets_df.loc[
        properties_on_streets_df.residential
    ]
    # the `roads_id` is repeated every time a property has it as nearest road
    property_counts_dict = dict(residential_properties_df.roads_id.value_counts())

    # we can now ignore the individual properties and focus on the streets
    streets = properties_on_streets_df[
        ~properties_on_streets_df.roads_geometry.duplicated()
    ][["roads_id", "startNode", "endNode", "roads_geometry"]]
    streets["number_of_properties"] = streets.roads_id.apply(
        lambda roads_id: property_counts_dict.get(roads_id, 0)
    )

    # convert to GeoDataFrame
    streets = gpd.GeoDataFrame(
        streets, geometry=gpd.GeoSeries.from_wkb(streets.roads_geometry)
    )
    streets["length_per_property"] = (
        streets.geometry.length / streets.number_of_properties
    )
    streets["residential"] = streets.length_per_property < residential_street_threshold

    return (
        streets,
        gpd.GeoDataFrame(
            residential_properties_df,
            geometry=gpd.GeoSeries.from_wkb(
                residential_properties_df.properties_geometry
            ),
        ),
    )


def get_properties_on_streets(db: data.Base, polygon: Polygon):
    """
    Use the data.Base k-nearest-neighbours method .knn() to get a
    dataframe of all properties in the given arealpolygon, each with their
    nearest road.
    """

    return db.knn(
        "properties", "roads", polygon, t2_columns=['"startNode"', '"endNode"']
    )


def mark_residential_properties(properties, db, polygon):
    """
    A property point is marked as residential if it is in a building
    polygon marked as residential.
    """

    df = properties.copy()
    # we say a property is residential if it is in a building
    properties_in_buildings = get_properties_in_buildings(db, polygon)

    properties_in_buildings["residential"] = mark_residential_buildings(
        properties_in_buildings
    )

    df["residential"] = df.properties_id.apply(
        lambda properties_id: properties_id
        in properties_in_buildings.loc[
            properties_in_buildings.residential
        ].properties_id.values
    )
    return df.residential


def get_properties_in_buildings(db, polygon):
    """
    Use the data.Base k-nearest-neighbours method .knn()
    to get all properties in the given areal polygon, together with
    their nearest building polygon.

    Then drop all buildings for which the distance between the property
    point and the building is not zero, ie. for which the building
    polygon does not include the property point.
    """

    # query db for nearest building for each property in polygon
    df = db.knn("properties", "buildings", polygon)

    # if a property is in a building, then the distance
    # between it and its nearest building is 0
    return df.loc[df.dist == 0]


def mark_residential_buildings(properties_in_buildings_df):
    """
    Mark building polygons as residential (or not).

    """

    df = properties_in_buildings_df.copy()

    # TODO: write logic to distinguish residential buildings
    # idea 1. footprint per property must be under threshold.
    # idea 2. institutional buildings will be labelled by `names`.

    # but for the moment we'll just say that for everything
    df["residential_building"] = True

    return df.residential_building


def snap_together_nearby_nodes(streets_df, max_node_distance, db, polygon):
    """
    Snap together nearby start- and end-nodes of streets so that they
    are considered as the same node and we can treat their streets as
    connected.

    I do this by using a `translator` that renames nodes as necessary.
    """

    # first we need the translator
    translator = get_nearest_nodes_translator(db, polygon, max_node_distance)
    print(f"There are {len(translator)} nearby nodes to be snapped together...")

    # then we can use it to rename the node
    streets_df["startNode"] = streets_df.startNode.apply(lambda x: translator.get(x, x))
    streets_df["endNode"] = streets_df.endNode.apply(lambda x: translator.get(x, x))

    return streets_df


def get_nearest_nodes_translator(db, polygon, max_node_distance):
    """
    Get a `dict` translator to rename nearby nodes to be 'snapped'
    together.
    """

    # get nodes nearby to each other
    nearest_nodes_df = db.knn("nodes", "nodes", polygon, k=5)

    # restrict to ones within max_distance
    nearest_nodes_df = nearest_nodes_df.loc[nearest_nodes_df.dist < max_node_distance]

    # the only columns we need are the ids, which
    # (because of the way I've written `db.knn()`)
    # both have the same name, so we need to rename them
    nearest_nodes_df = nearest_nodes_df[["nodes_id"]]
    assert len(nearest_nodes_df.columns) == 2
    nearest_nodes_df.columns = ["first", "second"]

    g = nx.from_pandas_edgelist(nearest_nodes_df, "first", "second")
    subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
    translator = {n: list(sorted(g.nodes))[0] for g in subgraphs for n in g.nodes}
    return translator


def label_connected_communities(all_streets_df, properties_df):
    """
    Label connected communities.
    """

    # we want to find connected communities of streets that are short or residential
    streets_df = all_streets_df.loc[all_streets_df.short_or_residential]
    street_graph = nx.from_pandas_edgelist(streets_df, "startNode", "endNode", True)
    subgraphs = [
        street_graph.subgraph(component)
        for component in nx.connected_components(street_graph)
    ]

    communities = {
        str(i + 1).zfill(3): list(nx.get_edge_attributes(graph, "roads_id").values())
        for i, graph in enumerate(subgraphs)
    }

    communities_key = {
        value: key for key, value_list in communities.items() for value in value_list
    }
    all_streets_df["community"] = all_streets_df.roads_id.apply(
        lambda street_id: communities_key.get(street_id, None)
    )
    properties_df["community"] = properties_df.roads_id.apply(
        lambda street_id: communities_key.get(street_id, None)
    )

    return all_streets_df, properties_df


def plot_postcode_pixel(
    postcode: str, min_community_size=5, ax=None, db: data.Base = None
):
    """
    Plot connected communities in pixel containing given postcode.
    """

    if not db:
        db = data.Base()

    postcode_point = db.query(
        f"SELECT geometry FROM names WHERE name1='{postcode.upper()}'", True
    ).geometry[0]
    pixel = db.intersects("pixels", postcode_point).geometry[0]
    streets, properties = get_communities(db, pixel, quiet=False)

    if not ax:
        _, ax = plt.subplots(figsize=(20, 20))

    streets.plot(ax=ax, color="grey")
    streets.loc[streets.community_size >= min_community_size].plot(
        "community", ax=ax, linewidth=2.5, linestyle="--", legend=True, cmap="tab20"
    )
    properties.plot(color="k", ax=ax, markersize=2)
    properties.loc[properties.faceblock_community_size >= min_community_size].plot(
        "community", ax=ax, markersize=1, cmap="tab20"
    )

    return ax.figure
