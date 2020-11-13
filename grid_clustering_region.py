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
import time

REGION_FILTER_DICT = {
    1: "='SUMBAGUT'",
    2: "='SUMBAGSEL'",
    3: "like '%%JABOTABEK'",
    4: "='JABAR'",
    5: "='JATENG-DIY'",
    6: "='JATIM'",
    7: "='BALI NUSRA'",
    8: "='KALIMANTAN'",
    9: "='SULAWESI'",
    10: "='MALUKU DAN PAPUA'",
    11: "='SUMBAGTENG'"

}

KPI_PERIOD_DICT ={
    # 'monthly' : ['initial_buffering', 'video_throughput', 'web_loading_time', 'rebuffering'],
    'weekly' : ['coverage', 'quality', 'download', 'upload', 'latency']
}


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
    """ converting epsilon from KM distance to radians"""
    return (epsilon - dist_matrix.min()) / (dist_matrix.max() - dist_matrix.min())


def dist_weight(X):
    """X should be the feature representations of shape (n_sample, dim_embeddings)"""
    return euclidean_distances(X)


def clusterize_region(date_id, period, model, tech, id_reg):
    print(date_id, period, model, tech, id_reg)
    t0 = time.time()
    ntp_engine_root = create_engine('postgresql://postgres:Immsp4102@10.53.205.5:5432/dna')
    ntp_engine = create_engine('postgresql://ntp_user:ntp#123@10.53.205.5:5432/dna')
    nea_engine = create_engine('postgresql://postgres:Immsp4102@10.53.205.5:5432/neadb')


    param_query = """
        select * from dna_data.t_clustering_lookup 
        where name = '{}' and data_unit='grid' and tech = {}
    """

    print(param_query.format(model, tech))
    df_params = pd.read_sql(sql=param_query.format(model, tech), con=ntp_engine)
    params = df_params.iloc[0]

    tech = params['tech']
    data_source = params['data_source']
    data_unit = params['data_unit']
    kpi_column = params['kpi_column']
    is_low_bad = params['is_low_bad']
    thresholds = params[[i for i in params.index if i[:4] == 'thd_']].to_list()
    alpha = params['alpha']
    beta = 1 - alpha
    min_radius = params['min_radius']
    print(params)

    if model in ['initial_buffering', 'video_throughput', 'web_loading_time', 'rebuffering']:
        if model == 'web_loading_time':
            raw_table = 'mqa_web_raw_location'
        else:
            raw_table = 'mqa_video_raw_location'
        raw_query = """
            with p as (select
                ST_GeoHash(ST_MakePoint(gps_lon::float8,gps_lat::float8),7) as geohash7,
                avg({3}::float8) as kpi
            from neadump.""" + raw_table + """_{0}
            where "ID_REG" = {1}
                and rad_mcc_end::int = 510
                and rad_mnc_end::int = 10
                and "agg_bearer_dim group_label" = '{2}G'
            group by 1
            ) select             
                st_x(ST_PointFromGeoHash(geohash7)) as longitude, 
                st_y(ST_PointFromGeoHash(geohash7)) as latitude,
                * 
            from p
        """
        raw_query = raw_query.format(date_id, id_reg, tech, kpi_column)
        kpi_engine = nea_engine
    else:
        raw_query = """
        select 
            longitude, latitude, geohash7, max({3}) as kpi
        from tutela.tutela_grid_qq_{0}
        where region {1}
            and  operator = 'Telkomsel' 
            and node = '{2}' and {3} is not null
        group by 1,2,3
        """
        kpi_engine = ntp_engine_root
        filter_region = REGION_FILTER_DICT[id_reg]
        if tech == 4:
            tech_str = '4G'
        else:
            tech_str = 'OTHERS'
        raw_query = raw_query.format(date_id, filter_region, tech_str, kpi_column)

    print(raw_query)
    df_raw = pd.read_sql(sql=raw_query, con=kpi_engine).dropna()
    print('raw_data len:'+ str(len(df_raw)))

    df = df_raw
    weight_col_raw = 'kpi'
    weight_col = weight_col_raw + '_flag'

    long_col = 'longitude'
    lat_col = 'latitude'

    df, thd_bad = discretize_wegiht(df, weight_col_raw, weight_col, thresholds, is_low_bad, is_thd_dynamic=False)

    if is_low_bad:
        print('bad data len: ', len(df[df[weight_col_raw] < thd_bad]))
        if (len(df[df[weight_col_raw] < thd_bad])) == 0:
            print('no bad data')
            return
    else:
        print('bad data len: ', len(df[df[weight_col_raw] > thd_bad]))
        if (len(df[df[weight_col_raw] > thd_bad])) == 0:
            print('no bad data')
            return

    df_geo = df.dropna()
    df_geo = df_geo[['geohash7', long_col, lat_col, weight_col, weight_col_raw, ]].dropna()
    df_geo_good = df_geo[df_geo[weight_col] != 3]
    df_geo = df_geo[df_geo[weight_col] == 3]
    dist_matrix_geo_raw = dist_geo(df_geo[[lat_col, long_col]])

    eps = min_radius
    min_samples = 2

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
    db.fit_predict(dist_matrix_geo_raw)

    labels = db.labels_
    clusters, counts = np.unique(labels, return_counts=True)
    df_geo['cluster'] = [str(x) for x in labels]
    print('number of cluster:, ', len(clusters))

    df_cluster = df_geo[(df_geo['cluster'] != '-1')]

    if is_low_bad:
        df_cluster = df_cluster.groupby('cluster').filter(lambda x: x[weight_col_raw].mean() <= thd_bad)
    else:
        df_cluster = df_cluster.groupby('cluster').filter(lambda x: x[weight_col_raw].mean() >= thd_bad)


    chulls = []
    df_cluster_agg = df_cluster.groupby('cluster').agg({weight_col_raw: ['mean']})
    df_cluster_agg.columns = df_cluster_agg.columns.droplevel(1)
    for i, cluster in enumerate(df_cluster_agg.index):
        points = df_cluster.loc[df_cluster['cluster'] == cluster, [long_col, lat_col]].drop_duplicates().values
        ghs = []
        for point in points:
            ghs.append(geohash_to_polygon(geohash.encode(point[1], point[0], 7)))
        chulls.append(cascaded_union(ghs))

    gdf_cluster = GeoDataFrame(df_cluster_agg, geometry=chulls)


    cluster_table_name = 'cluster_grid_' + str(tech) + 'g_' + period+ '_' + str(date_id)
    grid_table_name = 'bad_grid_' + str(tech) + 'g_' + period + '_' + str(date_id)

    database = psycopg2.connect(host='10.53.205.5',
                                port=5432,
                                user='ntp_user',
                                password='ntp#123',
                                database='dna')
    cursor = database.cursor()

    cursor.execute(
        "create table if not exists cluster_result." + cluster_table_name + "() inherits (cluster_result.cluster_grid);")
    cursor.execute(
        "create table if not exists cluster_result." + grid_table_name + "() inherits (cluster_result.bad_grid);")
    database.commit()

    cursor = database.cursor()
    cursor.execute("delete from cluster_result." + grid_table_name + " where id_reg=" + str(
        id_reg) + " and kpi_name = '" + model + "';")
    cursor.execute("delete from cluster_result." + cluster_table_name + " where id_reg=" + str(
        id_reg) + " and kpi_name = '" + model + "';")
    database.commit()

    cursor.close()
    database.close()

    df_pg = pd.DataFrame(df_cluster_agg.index, columns=['cluster'])
    df_pg['kpi_value'] = [i for i in df_cluster_agg[weight_col_raw]]
    df_pg['geom'] = [str(binascii.hexlify(i.wkb)).replace("b'", "").replace("'", "") for i in gdf_cluster['geometry']]
    df_pg['kpi_name'] = model
    df_pg['tech'] = tech
    df_pg['period'] = period
    df_pg['date_id'] = date_id
    df_pg['id_reg'] = id_reg

    df_geo_good['cluster'] = None
    df_geo_good = df_geo_good.drop(weight_col, 1).rename(columns={'kpi': 'kpi_value'})
    df_grid = df_geo.copy().drop(weight_col, 1).rename(columns={'kpi': 'kpi_value'})
    df_grid = pd.concat([df_grid, df_geo_good])
    df_grid['kpi_name'] = model
    df_grid['tech'] = tech
    df_grid['period'] = period
    df_grid['date_id'] = date_id
    df_grid['id_reg'] = id_reg

    print('writing to db')
    df_pg.to_sql(cluster_table_name,
                 con=ntp_engine,
                 schema='cluster_result',
                 if_exists='append',
                 index=False)
    df_grid.to_sql(grid_table_name,
                   con=ntp_engine,
                   schema='cluster_result',
                   if_exists='append',
                   index=False)
    print('done, time:', time.time()-t0)

def parseArgs():
    parser = argparse.ArgumentParser(
        description='Parse inputs for grid clustering DNA.')
    parser.add_argument('--period', action="store", dest="period",
                        help="period",
                        required=True)
    parser.add_argument('--date', action="store", dest="date_id",
                        help="date at which DNA clustering will be performed.",
                        required=True)

    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    period = args.period
    date_id = args.date_id
    for tech in [4,3,2]:
        if (period == 'monthly') and tech == 2:
            continue
        for i in range(1,12):
            for k in KPI_PERIOD_DICT[period]:
                clusterize_region(date_id, period, k, tech, i)



if __name__ == "__main__":
    main()