import pandas as pd
from geopandas import GeoDataFrame
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import haversine_distances, cosine_distances, euclidean_distances
from shapely.ops import cascaded_union
from polygon_geohasher.polygon_geohasher import geohash_to_polygon
import geohash
from sqlalchemy import create_engine
import argparse
import psycopg2
import binascii
import alphashape
import shapely.geometry as gmt


# %%
def parseArgs():
    parser = argparse.ArgumentParser(
        description='Parse inputs for sample clustering DNA.')
    parser.add_argument('--date', action="store", dest="date_id",
                        help="date at which DNA clustering will be performed.",
                        required=True)
    parser.add_argument('--id_kab', action="store", dest="id_kab",
                        help="id kabupaten.",
                        required=True)
    parser.add_argument('--kpi_name', action="store", dest="kpi",
                        help="KPI",
                        required=True)
    parser.add_argument('--tech', action="store", dest="tech",
                        help="tech", default=None,
                        required=True)

    args = parser.parse_args()
    return args


def get_zoom_level(df_raw, longitude_column, latitude_column, offset_ratio):
    df = (df_raw[[longitude_column, latitude_column]]).dropna()
    h_len = df[latitude_column].max() - df[latitude_column].min()
    v_len = df[longitude_column].max() - df[longitude_column].min()

    y_min = df[latitude_column].min() - (v_len * offset_ratio)
    y_max = df[latitude_column].max() + (v_len * offset_ratio)
    x_min = df[longitude_column].min() - (h_len * offset_ratio)
    x_max = df[longitude_column].max() + (h_len * offset_ratio)

    return x_min, x_max, y_min, y_max


def discretize_wegiht(df, weight_col, new_weight_col, thresholds, is_low_bad, is_thd_dynamic=False):
    range = df[weight_col].max() - df[weight_col].min()

    thd_bad, thd_fair, thd_good = thresholds[:3]

    df[new_weight_col] = 0

    thd_high = (df[weight_col].quantile(.5))
    thd_low = (df[weight_col].quantile(.25))

    if is_thd_dynamic:
        if is_low_bad:
            df.loc[df[weight_col] < thd_high, new_weight_col] = 1
            df.loc[df[weight_col] < thd_low, new_weight_col] = 2
            return df, thd_high
        else:
            df.loc[df[weight_col] > thd_high, new_weight_col] = 1
            df.loc[df[weight_col] > thd_low, new_weight_col] = 2
            return df, thd_low
    else:
        if is_low_bad:
            df.loc[df[weight_col] < thd_good, new_weight_col] = 1
            df.loc[df[weight_col] < thd_fair, new_weight_col] = 2
            df.loc[df[weight_col] < thd_bad, new_weight_col] = 3
        else:
            df.loc[df[weight_col] > thd_good, new_weight_col] = 1
            df.loc[df[weight_col] > thd_fair, new_weight_col] = 2
            df.loc[df[weight_col] > thd_bad, new_weight_col] = 3
        return df, thd_bad


def dist_geo(X):
    """Geo distance. X and Y should be lat/lon of shape (n_sample, 2)"""
    X_in_radians = np.radians(X)
    dist = haversine_distances(X_in_radians)
    dist *= 6371.0
    return dist


def _normalize_dist_matrix(dist_matrix):
    """MinMax scaling of distances in [0,1]"""
    return (dist_matrix - dist_matrix.min()) / (dist_matrix.max() - dist_matrix.min())


def normalize_epsilon(epsilon, dist_matrix):
    return (epsilon - dist_matrix.min()) / (dist_matrix.max() - dist_matrix.min())


def dist_weight(X):
    """X should be the feature representations of shape (n_sample, dim_embeddings)"""
    return euclidean_distances(X)


args = parseArgs()
model = args.kpi
date_id = args.date_id
id_kab = args.id_kab
tech = args.tech

param_query = """
    select * from dna_data.t_clustering_lookup 
    where name = '{}' and data_unit='sample' and tech = {}
"""

ntp_engine_root = create_engine('postgresql://postgres:Immsp4102@10.53.205.5:5432/dna')
ntp_engine = create_engine('postgresql://ntp_user:ntp#123@10.53.205.5:5432/dna')
nea_engine = create_engine('postgresql://postgres:Immsp4102@10.53.205.5:5432/neadb')

df_params = pd.read_sql(sql=param_query.format(model, tech), con=ntp_engine)
params = df_params.iloc[0]

tech = params['tech']
data_source = params['data_source']
data_unit = params['data_unit']
kpi_column = params['kpi_column']
kpi_table = params['kpi_table']
is_low_bad = params['is_low_bad']
thresholds = params[[i for i in params.index if i[:4] == 'thd_']].to_list()
alpha = params['alpha']
beta = 1 - alpha
min_radius = params['min_radius']

if model not in ['download', 'upload']:
    if model == 'web_loading_time':
        raw_table = 'mqa_web_daily_raw'
    else :
        raw_table = 'mqa_video_raw_location'
    raw_query = """
        select
            gps_lon::float8 as longitude,
            gps_lat::float8 as latitude,
            {3}::float8 as kpi
        from neadump.{4}_{0}
        where "ID_KAB" = {1}
        	and rad_mcc_end::int = 510
	        and rad_mnc_end::int = 10
            and "agg_bearer_dim group_label" = '{2}G'
    """.format(date_id, id_kab, tech, kpi_column, kpi_table,)


else:
    raw_query = """
    select *,
        gps_lon::float8 as longitude, 
        gps_lat::float8 as latitude, 
        {3}::float8 as kpi
    from neadump.{4}_{0}
    where "ID_KAB" = {1}
        and direction = '{5}'
        and "agg_bearer_dim group_label" = '{2}G' 
    """.format(date_id, id_kab, tech, kpi_column, kpi_table, model.title())

print(raw_query)
df_raw = pd.read_sql(sql=raw_query, con=nea_engine).dropna()
print(df_raw)

df = df_raw
weight_col_raw = 'kpi'
weight_col = weight_col_raw + '_flag'

long_col = 'longitude'
lat_col = 'latitude'

df, thd_bad = discretize_wegiht(df, weight_col_raw, weight_col, thresholds, is_low_bad, is_thd_dynamic=False)
print(thd_bad)

if is_low_bad:
    print(len(df[df[weight_col_raw] < thd_bad]))
    if (len(df[df[weight_col_raw] < thd_bad])) == 0:
        print('no bad data')
        exit()
else:
    print(len(df[df[weight_col_raw] > thd_bad]))
    if (len(df[df[weight_col_raw] > thd_bad])) == 0:
        print('no bad data')
        exit()

df_geo = df.dropna()
df_geo = df_geo[[long_col, lat_col, weight_col, weight_col_raw]].dropna()
dist_matrix_geo_raw = dist_geo(df_geo[[lat_col, long_col]])

max_dist = dist_matrix_geo_raw.max()
dist_matrix_geo = _normalize_dist_matrix(dist_matrix_geo_raw)

eps = normalize_epsilon(min_radius, dist_matrix_geo_raw)
print(eps)
min_samples = 5

dist_matrix_weight = _normalize_dist_matrix(dist_weight(df_geo[[weight_col]]))
dist_matrix = alpha * dist_matrix_geo + beta * dist_matrix_weight
db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
db.fit_predict(dist_matrix)

labels = db.labels_
clusters, counts = np.unique(labels, return_counts=True)
df_geo['cluster'] = [str(x) for x in labels]

df_cluster = df_geo[(df_geo['cluster'] != '-1')]

if is_low_bad:
    print('low')
    df_cluster = df_cluster.groupby('cluster').filter(lambda x: x[weight_col_raw].mean() <= thd_bad)
else:
    print('high')
    df_cluster = df_cluster.groupby('cluster').filter(lambda x: x[weight_col_raw].mean() >= thd_bad)
print(thd_bad)

chulls = []
df_cluster_agg = df_cluster.groupby('cluster').agg({weight_col_raw: ['median']})
df_cluster_agg.columns = df_cluster_agg.columns.droplevel(1)

for i, cluster in enumerate(df_cluster_agg.index):
    points = df_cluster.loc[df_cluster['cluster'] == cluster, [long_col,lat_col]].drop_duplicates().values
    try:
        hull = alphashape.alphashape(points, 375)
        if hull.type != 'Polygon':
            ghs = []
            for point in points:
                ghs.append(geohash_to_polygon(geohash.encode(point[1],point[0],7)))
            points = []
            for polygon in ghs:
                points.extend(polygon.exterior.coords[:-1])
            hull = alphashape.alphashape(points,375)
#         hull = alphashape.alphashape(ghs, 375)
    except Exception:
        ghs = []
        for point in points:
            ghs.append(geohash_to_polygon(geohash.encode(point[1],point[0],7)))
        points = []
        for polygon in ghs:
            points.extend(polygon.exterior.coords[:-1])
        hull = alphashape.alphashape(points,375)
    chulls.append(hull)

gdf_cluster = GeoDataFrame(df_cluster_agg, geometry=chulls)

database = psycopg2.connect(host='10.53.205.5',
                            port=5432,
                            user='ntp_user',
                            password='ntp#123',
                            database='dna')
cursor = database.cursor()
cursor.execute('create table if not exists cluster_result.cluster_sample_' + model + '_' + str(tech)  + 'g_' + str(date_id) + "() inherits (cluster_result.cluster_sample);")
cursor.execute('delete from cluster_result.cluster_sample_' + model + '_' + str(tech)  + 'g_' + str(date_id) + " where id_kab="+str(id_kab)+";")
database.commit()
cursor.close()
database.close()

df_pg = pd.DataFrame(df_cluster_agg.index, columns=['cluster'])
df_pg['kpi_value'] = [i for i in df_cluster_agg[weight_col_raw]]
df_pg['geom'] = [str(binascii.hexlify(i.wkb)).replace("b'","").replace("'","") for i in gdf_cluster['geometry']]
df_pg['kpi_name'] = model
df_pg['tech'] = tech
df_pg['date_id'] = date_id
df_pg['id_kab'] = id_kab

result_table_name= 'cluster_sample_' + model + '_' + str(tech)  + 'g_' + str(date_id)
print(result_table_name)

df_pg.to_sql(result_table_name,
             con=ntp_engine,
             schema='cluster_result',
             if_exists='append',
             index=False)